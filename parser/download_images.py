import asyncio
import aiohttp
import aiofiles
import os


async def write_file(session, category, url, name_img):
    async with aiofiles.open(f'../data/images/{category}/{name_img}.jpg', mode='wb') as file:
        async with session.get(url) as response:
            async for x in response.content.iter_chunked(1024):
                await file.write(x)


async def main(category, links):
    async with aiohttp.ClientSession() as session:

        tasks = []
        for link in links:
            name = hash(link)
            task = asyncio.create_task(write_file(session, category, link, name))
            tasks.append(task)
        await asyncio.gather(*tasks)


def get_new_images(category):
    with open(f'../data/links/{category}.txt', 'r') as file_:
        urls = file_.read().split('\n')
    os.mkdir(f'../data/images/{category}')
    asyncio.run(main(category, urls))


if __name__ == '__main__':
    cat = input()
    get_new_images(cat)
