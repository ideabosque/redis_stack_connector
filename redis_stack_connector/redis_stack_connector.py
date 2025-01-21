#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

import logging
import struct
import traceback
from typing import Any, Dict, List, Tuple

import redis
from redis.commands.search.field import NumericField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

# Constants
VECTOR_DIM = 1536  # Dimension for embeddings
DISTANCE_METRIC = "COSINE"  # Distance metric for vector similarity


class RedisStackConnector:
    def __init__(self, logger: logging.Logger, **setting: Dict[str, Any]):
        self.logger = logger
        self.redis_client = redis.Redis(
            host=setting["REDIS_HOST"],
            port=setting["REDIS_PORT"],
            password=setting["REDIS_PASSWORD"],
        )
        self.setting = setting

    def index_exists(self, index_name: str):
        """
        Determines whether the specified Rediscript index exists
        :Param index_name: index name
        :Return: True if the index exists, otherwise False
        """
        try:
            if self.redis_client.ft(index_name).info():
                return True
            return False
        except ResponseError as e:
            if "unknown index name" in str(e).lower():
                return False
            raise

    def create_redis_index(self, index_name: str, fields: Dict[str, Any], prefix: str):
        try:
            # Check if the index already exists
            if self.index_exists(index_name=index_name):
                print(f"Index '{index_name}' already exists.")
                return

            index_fields = []

            for field_name, field_type in fields.items():
                if field_type == "TEXT":
                    index_fields.append(TextField(field_name))
                elif field_type == "VECTOR":
                    index_fields.append(
                        VectorField(
                            field_name,
                            "HNSW",
                            {
                                "TYPE": "FLOAT32",
                                "DIM": VECTOR_DIM,
                                "DISTANCE_METRIC": DISTANCE_METRIC,
                            },
                        )
                    )
                elif field_type == "NUMERIC":
                    index_fields.append(NumericField(field_name))
                else:
                    raise Exception(f"Invalid field type: {field_type}")

            self.redis_client.ft(index_name).create_index(
                fields=index_fields,
                definition=IndexDefinition(prefix=[prefix], index_type=IndexType.HASH),
            )
            return

        except Exception as e:
            print(
                "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
            )
            log = traceback.format_exc()
            self.logger.error(log)
            raise e

    def index_document(self, prefix: str, key: str, doc: Dict[str, Any]):
        try:
            key = f"{prefix}:{str(doc[key])}"

            # Convert list of floats to byte vectors for title
            if doc.get("title_vector"):
                title_embedding = struct.pack(
                    f"{len(doc['title_vector'])}f", *doc["title_vector"]
                )
                doc["title_vector"] = title_embedding

            # Convert list of floats to byte vectors for content
            content_embedding = struct.pack(
                f"{len(doc['content_vector'])}f", *doc["content_vector"]
            )
            doc["content_vector"] = content_embedding

            self.redis_client.hset(key, mapping=doc)

            return

        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise e

    def search_redis(
        self,
        query_vector: List[Dict[str, Any]],
        index_name: str,
        vector_field: str = None,
        return_fields: list = None,
        hybrid_fields: str = "*",
        k: int = 100,
        offset: int = 0,
        limit: int = 100,
        use_ann: bool = False,  # Enable ANN (HNSW)
        ef_runtime: int = 10,  # Efficiency factor for HNSW (only used if ANN is enabled)
    ) -> Tuple[int, List[Dict[str, Any]]]:
        try:
            if vector_field is None:
                vector_field = (
                    self.setting.get("redis_index_config", {})
                    .get(index_name, {})
                    .get("vector_field")
                )
            if return_fields is None:
                return_fields = (
                    self.setting.get("redis_index_config", {})
                    .get(index_name, {})
                    .get("return_fields")
                )

            # Prepare the Query
            if use_ann:
                # For HNSW (ANN), include `EF_RUNTIME`
                base_query = f"{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score EF_RUNTIME {ef_runtime}]"
            else:
                # Default KNN query
                base_query = f"{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]"

            query = (
                Query(base_query)
                .sort_by("vector_score")
                .paging(offset, limit)
                .dialect(2)
            )
            if return_fields:
                query.return_fields(*return_fields)

            # Convert embedding to bytes
            params_dict = {
                "vector": struct.pack(f"{len(query_vector)}f", *query_vector)
            }

            # Perform vector search
            result = self.redis_client.ft(index_name).search(query, params_dict)
            return result.total, result.docs

        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise e

    def search_vector(
        self,
        query_vector: List[Dict[str, Any]],
        index_name: str,
        **kwargs: Dict[str, Any],
    ) -> Tuple[int, List[Dict[str, Any]]]:
        try:
            _kwargs = {
                "vector_field": kwargs.get("vector_field"),
                "return_fields": kwargs.get("fields_to_return"),
                **{
                    mapped_key: kwargs[key]
                    for key, mapped_key in {
                        "filter_conditions": "hybrid_fields",
                        "top_k": "k",
                        "result_offset": "offset",
                        "limit": "limit",
                    }.items()
                    if key in kwargs
                },
                **(
                    {
                        "use_ann": self.setting["use_ann"],
                        "ef_runtime": self.setting.get("ef_runtime", 10),
                    }
                    if "use_ann" in self.setting
                    else {}
                ),
            }

            total, results = self.search_redis(
                query_vector,
                index_name,
                **_kwargs,
            )
            return total, results
        except Exception as e:
            self.logger.error(f"Error during vector search: {traceback.format_exc()}")
            raise e

    def create_hybrid_field(
        self, field_name: str, values: List[str], relationship_operator: str = "|"
    ) -> str:
        return f'@{field_name}:({" {} ".format(relationship_operator).join(values)})'
