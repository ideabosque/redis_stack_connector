#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

import logging
import struct
import traceback
from typing import Any, Dict, List

import redis
from openai import OpenAI
from redis.commands.search.field import NumericField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

# Constants
VECTOR_DIM = 1536  # Dimension for embeddings
DISTANCE_METRIC = "COSINE"  # Distance metric for vector similarity


class RedisStackConnector:
    def __init__(self, logger: logging.Logger, **setting: Dict[str, Any]):
        self.logger = logger
        self.openai_client = OpenAI(
            api_key=setting["openai_api_key"],
        )
        self.redis_client = redis.Redis(
            host=setting["REDIS_HOST"],
            port=setting["REDIS_PORT"],
            password=setting["REDIS_PASSWORD"],
        )
        self.embedding_model = setting["EMBEDDING_MODEL"]

    def create_redis_index(self, index_name: str, fields: Dict[str, Any], prefix: str):
        try:
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

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        res = self.openai_client.embeddings.create(
            input=[text], model=self.embedding_model
        )
        return res.data[0].embedding

    def search_redis(
        self,
        user_query: str,
        index_name: str,
        vector_field: str = "content_vector",
        return_fields: list = None,
        hybrid_fields: str = "*",
        k: int = 100,
        offset: int = 0,
        limit: int = 100,
        use_ann: bool = False,  # Enable ANN (HNSW)
        ef_runtime: int = 10,  # Efficiency factor for HNSW (only used if ANN is enabled)
    ) -> List[Dict[str, Any]]:
        try:
            # Creates embedding vector from user query
            embedded_query = self.get_embedding(user_query)

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
                "vector": struct.pack(f"{len(embedded_query)}f", *embedded_query)
            }

            # Perform vector search
            result = self.redis_client.ft(index_name).search(query, params_dict)
            return result.total, result.docs

        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise e

    def create_hybrid_field(
        self, field_name: str, values: List[str], relationship_operator: str = "|"
    ) -> str:
        return f'@{field_name}:({" {} ".format(relationship_operator).join(values)})'
