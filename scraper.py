import requests
from bs4 import BeautifulSoup

def get_latest_news():
    """Tìm kiếm tin tức mới nhất từ web."""
    url = "https://vnexpress.net"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    news = [item.text for item in soup.select(".title-news a")]
    return news[:5]  # Lấy 5 tin tức mới nhất

print("🔍 Tin tức mới nhất:", get_latest_news())
