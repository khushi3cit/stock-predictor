import requests

def check_connection(url="https://www.alphavantage.co", timeout=5):
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except requests.ConnectionError:
        return False