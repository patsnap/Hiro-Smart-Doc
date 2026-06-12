import logging

import aiohttp


class SingletonAiohttp:
    session: aiohttp.ClientSession
    logger = logging.getLogger(__name__)
    api_logger = logging.getLogger("api")

    @classmethod
    def init_session(cls) -> None:
        cls.session = aiohttp.ClientSession(raise_for_status=True)
        cls.logger.info("aiohttp ClientSession initialized.")

    @classmethod
    async def close_session(cls) -> None:
        await cls.session.close()
        cls.logger.info("aiohttp ClientSession closed.")
