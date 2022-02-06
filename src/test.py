import requests
import fire

def input(
    string='I love me some data'
):
    host = 'http://localhost:8080'
    obj = {
        'string': (None, string)
    }

    x = requests.post(host, files = obj)
    print(x.text)
    print('\n\n')


if __name__ == '__main__':
    fire.Fire(input)