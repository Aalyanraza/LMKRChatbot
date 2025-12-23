# Web Scraping Tools

import time
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from bs4 import BeautifulSoup
import requests
from langchain_core.tools import tool
import config
from utils import clean_text_content, save_to_file

def fetch_and_clean_body(url: str, depth=0) -> str:
    """
    Fetches a webpage using Selenium and returns cleaned body text.
    """
    if depth > 1:
        return ""
    
    print(f" 🖥️ Booting Headless Edge for: {url}")
    edge_options = EdgeOptions()
    edge_options.add_argument("--headless")
    edge_options.add_argument("--no-sandbox")
    
    driver = None
    try:
        driver = webdriver.Edge(options=edge_options)
        driver.get(url)
        time.sleep(config.SELENIUM_WAIT_TIME)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Cleanup tags
        for tag in soup(config.CLEANUP_TAGS):
            tag.decompose()
        
        body = soup.find('body')
        if body:
            return body.get_text(separator="\n")
        else:
            return soup.get_text(separator="\n")
    
    except Exception as e:
        print(f"❌ Selenium Error: {e}")
        return ""
    
    finally:
        if driver:
            driver.quit()

@tool
def scrape_careers_tool():
    """
    Scrapes the official LMKR careers page to retrieve live job openings, requirements, and application emails.
    """
    print(f"🕸️ Tool Triggered: Dynamically scraping {config.CAREERS_URL}...")
    
    raw_text = fetch_and_clean_body(config.CAREERS_URL)
    clean_text = clean_text_content(raw_text)
    
    save_to_file(f"SOURCE: {config.CAREERS_URL}\n\n{clean_text}", config.CAREERS_OUTPUT_FILE)
    
    return clean_text

@tool
def scrape_news_fast_tool():
    """
    Scrapes the LMKR announcements page using Requests + BS4 to retrieve the latest news and press releases.
    """
    print(f"🗞️ Tool Triggered: Fast scraping {config.NEWS_URL}...")
    
    headers = {
        "User-Agent": config.SELENIUM_USER_AGENT
    }
    
    try:
        response = requests.get(config.NEWS_URL, headers=headers, timeout=config.SCRAPE_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Cleanup irrelevant tags
        for tag in soup(config.CLEANUP_TAGS):
            tag.decompose()
        
        body = soup.find('body')
        clean_text = clean_text_content(body.get_text(separator="\n")) if body else ""
        
        save_to_file(f"SOURCE: {config.NEWS_URL}\n\n{clean_text}", config.NEWS_OUTPUT_FILE)
        
        return clean_text
    
    except Exception as e:
        print(f"❌ Fast Scrape Error: {e}")
        return ""
