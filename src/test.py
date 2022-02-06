import requests
import fire

def input(
    string='I love me some data'
):
    host = 'http://localhost:80'
    obj = {
        'string': string
    }

    x = requests.post(host, data = obj)
    print(x.text)
    print('\n\n')


if __name__ == '__main__':
    fire.Fire(input)