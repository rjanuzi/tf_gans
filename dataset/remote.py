from requests import get

from dataset import DATASET_RAW_IMGS_FOLDER

ISIC_API_URL = "https://isic-archive.com/api/v1/"


def get_remote_imgs_list(list_limit=0, offset=0):
    try:
        url = ISIC_API_URL + "image?limit=%d&offset=%d&detail=true" % (
            list_limit,
            offset,
        )
        return get(url).json()
    except:
        return []


def download_img(id, img_name):
    try:
        url = ISIC_API_URL + "image/%s/download?contentDisposition=inline" % id
        path = r"%s\%s.jpg" % (DATASET_RAW_IMGS_FOLDER, img_name)
        response = get(url)
        with open(path, "wb") as fp:
            fp.write(response.content)
            return True
    except:
        return False
