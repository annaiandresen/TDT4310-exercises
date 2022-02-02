from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from bs4 import BeautifulSoup

driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get('https://www.ntnu.edu/')

# Raw HTML
content = driver.page_source

# Parse the HTML in bs4
soup = BeautifulSoup(content, features="html.parser")

tabs = soup.find_all('div', attrs={"class": "card-body"})