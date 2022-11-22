import redis
import json

from app.config import REDIS_URI

# The default redis max_connections in a pool is 2 ** 31
# this is very large, so we don't have to worry about max pool connections as MongoDB when reuse redis client


DEFAULT_KEY_EXPIRE_TIME = 3600 * 24  # 1 day


class RedisCache:
    """
    Redis Client manager in singleton style.
    Each Redis Client object have a connection-pooling built-in.
    The use of database, collection depend on this built-in pooling.
    """
    __client = None

    @staticmethod
    def _info_check(redis_uri):
        if not redis_uri:
            raise ValueError("redis_uri is empty, check env var REDIS_URI or your get_client() input")

    @staticmethod
    def get_client(new_instant_bool=False, redis_uri=REDIS_URI) -> redis.Redis:
        """Singleton-style client"""
        if new_instant_bool:
            return redis.Redis().from_url(redis_uri)
        if RedisCache.__client is None:
            RedisCache._info_check(redis_uri)
            try:
                RedisCache.__client = redis.Redis().from_url(redis_uri)
            except Exception as e:
                raise Exception(f"Failed to create client, details: {e}")
        return RedisCache.__client

    @staticmethod
    def r_set(key: str, data: str, expire: int = DEFAULT_KEY_EXPIRE_TIME) -> bool:
        ok = False
        c = RedisCache.get_client()
        if c:
            if not data:
                ok = c.set(key, '', ex=expire)
            else:
                ok = c.set(key, data, ex=expire)
        return ok

    @staticmethod
    def r_get(key: str):
        data = None
        c = RedisCache.get_client()
        if c:
            try:
                data = c.get(key)
            except Exception as e:
                raise Exception(f"Failed to get value with key {key} from Redis, details: {e}")
        return data

    @staticmethod
    def r_del(key: str) -> bool:
        ok = False
        c = RedisCache.get_client()
        if c:
            try:
                c.delete(key)
                ok = True
            except Exception as e:
                raise Exception(f"Failed to delete value with key {key} from Redis, details: {e}")
        return ok

    @staticmethod
    def r_flush_all() -> bool:
        ok = False
        c = RedisCache.get_client()
        if c:
            try:
                c.flushall()
                ok = True
            except Exception as e:
                raise Exception(f"Failed to flush all db (Redis), details: {e}")
        return ok

    @staticmethod
    def copy_key(source_key, dest_key) -> bool:
        c = RedisCache.get_client()
        ok = c.copy(source_key, dest_key)
        return ok

    @staticmethod
    def clear_ns(ns) -> bool:
        """
        Clear keys which match pattern.
        :param ns: str, namespace i.e your:prefix
        :return: int, cleared keys
        """
        cache = redis.StrictRedis()
        chunk_size = 5000
        cursor = '0'
        ns_keys = ns + '*'
        while cursor != 0:
            cursor, keys = cache.scan(cursor=cursor, match=ns_keys, count=chunk_size)
            if keys:
                cache.delete(*keys)
        return True

    @staticmethod
    def exists(key) -> bool:
        """Check if a key is exists"""
        c = RedisCache.get_client()
        return bool(c.exists(key))

    # helper function
    @staticmethod
    def set_dict(key: str, data: dict, expire: int = DEFAULT_KEY_EXPIRE_TIME) -> bool:
        ok = False
        c = RedisCache.get_client()
        if c:
            if not data:
                ok = c.set(key, '', ex=expire)
            else:
                ok = c.set(key, json.dumps(data), ex=expire)
        return ok

    @staticmethod
    def get_dict(key: str):
        data = None
        c = RedisCache.get_client()
        if c:
            try:
                val = c.get(key)
                data = json.loads(val) if val else None
            except Exception as e:
                raise Exception(f"Failed to get value with key {key} from Redis, details: {e}")
        return data

    @staticmethod
    def get_file(key: str) -> bytes:
        return RedisCache.r_get(key)

    @staticmethod
    def push_file(key: str, data: bytes, expire: int = DEFAULT_KEY_EXPIRE_TIME) -> bool:
        if not data:
            raise TypeError('data argument must not be NoneType or empty')
        c = RedisCache.get_client()
        ok = c.set(key, data, ex=expire)
        return ok
