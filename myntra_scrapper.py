import time
import csv
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import re
import time

# Configuration
BASE_URL = "https://www.myntra.com/personal-care?f=Categories%3ALipstick"
MAX_PAGES = 5
OUTPUT_FILE = "myntra_lipstick_data.csv"

def setup_driver():
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Run in background
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    # User agent to mimic real browser
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    return driver

def get_breadcrumbs(driver):
    # Myntra breadcrumbs are usually at the top inside a 'breadcrumbs-container' or list
    try:
        breadcrumbs = driver.find_element(By.CLASS_NAME, "breadcrumbs-list").text
        # Format: Home / Personal Care / ...
        return breadcrumbs.replace("\n", " / ")
    except:
        return "Home / Personal Care / Lipstick (Default)"

def extract_price_int(price_text: str):
    """
    Convert Myntra price text like 'Rs. 1,499', '₹349', '1,299', 
    or 'Rs. 399 (50% OFF)' → integer 1499, 349, 1299, 399
    """
    if not price_text:
        return None
    
    # Keep only digits and commas
    cleaned = re.sub(r"[^\d,]", "", price_text)

    # Remove commas
    cleaned = cleaned.replace(",", "")

    # Return int if found, else None
    return int(cleaned) if cleaned.isdigit() else None

def scrape_myntra():
    driver = setup_driver()
    all_products = []
    
    try:
        for page in range(1, MAX_PAGES + 1):
            url = f"{BASE_URL}&p={page}"
            print(f"Scraping Page {page}: {url}")
            driver.get(url)
            
            # Wait for product grid to load
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "product-base"))
                )
                time.sleep(1)  # Additional wait for stability  # allow JS to populate rating nodes
            except:
                print("Timeout or no products found on this page.")
                break

            # Scroll to bottom to trigger lazy loading if necessary
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3) # Respectful delay

            # Get Breadcrumb (usually static per category page)
            current_breadcrumb = get_breadcrumbs(driver)

            # Find product cards
            products = driver.find_elements(By.CLASS_NAME, "product-base")
            
            for product in products:
                try:
                    # Extract details
                    # Note: Class names on Myntra are dynamic (e.g., product-brand, product-product). 
                    # You may need to inspect the page source to verify these classes if they change.
                    
                    brand = product.find_element(By.CLASS_NAME, "product-brand").text
                    name = product.find_element(By.CLASS_NAME, "product-product").text
                    
                    # --- PRICE (convert to int) ---
                    price_raw = None
                    try:
                        price_raw = product.find_element(By.CLASS_NAME, "product-discountedPrice").text
                    except:
                        try:
                            price_raw = product.find_element(By.CLASS_NAME, "product-price").text
                        except:
                            price_raw = None

                    price = extract_price_int(price_raw)

                            
                    # Link
                    link_tag = product.find_element(By.TAG_NAME, "a")
                    product_url = link_tag.get_attribute("href")

                    # ------- Robust rating + review extraction -------
                    rating_score = 0.0   # float or None
                    review_count = 0      # int

                    # small helper to clean numbers like "4.5 out of 5" -> "4.5" and "1,234" -> 1234
                    def extract_first_number(s: str):
                        if not s:
                            return None
                        # try to find a float like 4.5 first
                        m = re.search(r'(\d+\.\d+)', s)
                        if m:
                            return m.group(1)
                        # otherwise find integer with commas/spaces
                        m = re.search(r'(\d{1,3}(?:[,\s]\d{3})+|\d+)', s)
                        if m:
                            return m.group(1).replace(",", "").replace(" ", "")
                        return None

                    try:
                        # make sure element is visible (sometimes ratings lazy-load)
                        try:
                            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", product)
                            time.sleep(0.15)
                        except Exception:
                            pass

                        # 1) Try the classes you used (some pages still use these)
                        try:
                            rc = product.find_element(By.CLASS_NAME, "product-ratingsCount").text.strip()
                            rs = product.find_element(By.CLASS_NAME, "product-ratingsScore").text.strip()
                            # rating_score might be like "4.3" or "4.3/5"
                            rs_n = extract_first_number(rs)
                            rating_score = float(rs_n) if rs_n else None

                            rc_n = extract_first_number(rc)
                            review_count = int(rc_n) if rc_n and rc_n.isdigit() else 0
                        except Exception:
                            # 2) Try a few common alternative selectors used on Myntra tiles
                            alt_selectors = [
                                ("span.product-rating", "span.product-rating-count"),
                                ("span.rating", "a.rating-link"),
                                ("div.product-starRating", "a.reviews"),
                                ("a.reviews", "a.reviews"),  # sometimes both in same anchor
                                ("span.myntra-rating-count", "span.myntra-rating-score")
                            ]
                            found = False
                            for sel_score, sel_count in alt_selectors:
                                try:
                                    rs_el = product.find_element(By.CSS_SELECTOR, sel_score)
                                    rc_el = product.find_element(By.CSS_SELECTOR, sel_count)
                                    rs_text = rs_el.text.strip()
                                    rc_text = rc_el.text.strip()
                                    rs_n = extract_first_number(rs_text)
                                    rc_n = extract_first_number(rc_text)
                                    if rs_n:
                                        rating_score = float(rs_n)
                                    if rc_n and rc_n.isdigit():
                                        review_count = int(rc_n)
                                    found = True
                                    break
                                except Exception:
                                    continue

                            if not found:
                                # 3) As a last resort search the product innerText for patterns
                                inner = product.get_attribute("innerText") or ""
                                # Look for ratings pattern like "4.5" (first float) and for review counts like "1,234"
                                r_float = extract_first_number(inner)
                                if r_float and "." in r_float:
                                    try:
                                        rating_score = float(r_float)
                                    except:
                                        rating_score = None
                                # For review count, find patterns like "1234 ratings" or "1,234"
                                m = re.search(r'(\d{1,3}(?:[,\s]\d{3})+|\d+)\s*(?:ratings|reviews|review|RATINGS|REVIEWS)?', inner, flags=re.IGNORECASE)
                                if m:
                                    rc_val = m.group(1).replace(",", "").replace(" ", "")
                                    if rc_val.isdigit():
                                        review_count = int(rc_val)

                    except Exception as e:
                        # fallback defaults if anything breaks
                        rating_score = rating_score if rating_score is not None else None
                        review_count = review_count if review_count else 0


                    data = {
                        "Brand": brand,
                        "Product Name": name,
                        "Price": price,
                        "Rating Score": rating_score,
                        "Review Count": review_count,
                        "URL": product_url,
                        "Breadcrumb": current_breadcrumb
                    }
                    all_products.append(data)
                    
                except Exception as e:
                    # Skip incomplete products
                    continue
            
            print(f"Found {len(products)} products on page {page}.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()

    # Save to CSV
    import pandas as pd
    import csv

    if all_products:
        df = pd.DataFrame(all_products)

        # Drop duplicates based on Product Name + Brand
        df.drop_duplicates(subset=['Product Name', 'Brand'], keep='first', inplace=True)

        # Overwrite output file with cleaned data
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

        print(f"Successfully saved {len(df)} unique items to {OUTPUT_FILE}")
    else:
        print("No data scraped.")

if __name__ == "__main__":
    scrape_myntra()