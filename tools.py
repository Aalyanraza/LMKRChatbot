# Web Scraping Tools
import os
import subprocess
import time
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeService # Add this import
from bs4 import BeautifulSoup
import requests
from langchain_core.tools import tool
import config
from utils import clean_text_content, save_to_file


def fetch_and_clean_body(url: str, depth=0) -> str:
    if depth > 1:
        return ""
    
    print(f" 🖥️ Booting Headless Edge for: {url}")
    
    # 1. Configure Options
    edge_options = EdgeOptions()
    edge_options.add_argument("--headless=new") 
    edge_options.add_argument("--no-sandbox")
    edge_options.add_argument("--log-level=3")
    edge_options.add_argument("--silent")
    edge_options.add_argument("--disable-gpu") 
    edge_options.add_argument("--disable-software-rasterizer")
    edge_options.add_argument("--disable-dev-shm-usage")
    edge_options.add_argument("--remote-debugging-port=0")
    
    # Crucial: This experimental option kills the DevTools logging
    edge_options.add_experimental_option('excludeSwitches', ['enable-logging', 'enable-automation'])

    # 2. Configure Service (This kills the 'LoadEnclaveImageW' and renderer noise)
    # We create a service that hides the window and pipes logs to nowhere
    edge_service = EdgeService()
    if os.name == 'nt': # Windows only flag
        edge_service.creation_flags = subprocess.CREATE_NO_WINDOW
    
    driver = None
    try:
        driver = webdriver.Edge(options=edge_options, service=edge_service)
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
