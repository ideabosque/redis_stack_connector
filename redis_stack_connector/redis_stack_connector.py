#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

import logging
import traceback
from typing import Any, Dict, List

import numpy as np
import redis
from openai import OpenAI
from redis.commands.search.query import Query


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

    def index_document(self, prefix: str, key: str, doc: Dict[str, Any]):
        try:
            key = f"{prefix}:{str(doc[key])}"

            # create byte vectors for title
            if doc.get("title_vector"):
                title_embedding = np.array(
                    doc["title_vector"], dtype=np.float32
                ).tobytes()

                # replace list of floats with byte vectors
                doc["title_vector"] = title_embedding

            # create byte vectors for content
            content_embedding = np.array(
                doc["content_vector"], dtype=np.float32
            ).tobytes()

            # replace list of floats with byte vectors
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

    def inquiry_data(self, **arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        kwargs = {"user_query": arguments["user_query"], "k": 10}
        if arguments.get("index_name"):
            kwargs["index_name"] = arguments["index_name"]
        if arguments.get("vector_field"):
            kwargs["vector_field"] = arguments["vector_field"]
        if arguments.get("return_fields"):
            kwargs["return_fields"] = arguments["return_fields"]
        if arguments.get("hybrid_fields"):
            kwargs["hybrid_fields"] = arguments["hybrid_fields"]
        if arguments.get("k"):
            kwargs["k"] = arguments["k"]
        return self.search_redis(**kwargs)

    def search_redis(
        self,
        user_query: str,
        index_name: str,
        vector_field: str = "content_vector",
        return_fields: list = None,
        hybrid_fields="*",
        k: int = 20,
    ) -> List[Dict[str, Any]]:
        try:
            # Creates embedding vector from user query
            embedded_query = self.get_embedding(user_query)

            # Prepare the Query
            base_query = (
                f"{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]"
            )
            query = Query(base_query).sort_by("vector_score").paging(0, k).dialect(2)
            if return_fields:
                query.return_fields(*return_fields)

            params_dict = {
                "vector": np.array(embedded_query).astype(dtype=np.float32).tobytes()
            }

            # perform vector search
            result = self.redis_client.ft(index_name).search(query, params_dict)
            return result.docs

        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise e

    def create_hybrid_field(
        self, field_name: str, values: List[str], relationship_operator: str = "|"
    ) -> str:
        return f'@{field_name}:({" {} ".format(relationship_operator).join(values)})'
