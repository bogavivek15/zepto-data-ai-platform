import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd


# Main website
BASE_URL = "https://books.toscrape.com/"


# Categories we want
TARGET_CATEGORIES = [
    "Mystery",
    "Historical Fiction",
    "Romance",
    "Fiction"
]


# Store scraped books
books_list = []


# Request main website
response = requests.get(BASE_URL)

print("Main page status:", response.status_code)

# Create soup
soup = BeautifulSoup(response.content, "html.parser")


# Find category links
category_list = soup.find("ul", class_="nav-list")

category_links = category_list.find_all("a")


# Go through every category link
for link in category_links:

    category_name = link.text.strip()

    # Only scrape our selected categories
    if category_name not in TARGET_CATEGORIES:
        continue

    # Create full category URL
    category_url = urljoin(
        BASE_URL,
        link["href"]
    )

    print("\nStarting category:", category_name)

    # Start from first page
    current_url = category_url


    # Pagination
    while True:

        print("Scraping:", current_url)

        response = requests.get(current_url)

        # Check request
        if response.status_code != 200:
            print("Request failed:", response.status_code)
            break

        soup = BeautifulSoup(
            response.content,
            "html.parser"
        )


        # Category name
        category = soup.find("h1").text.strip()


        # Find books
        products = soup.find_all(
            "article",
            class_="product_pod"
        )


        # Scrape books
        for product in products:

            title = product.find(
                "h3"
            ).find("a")["title"]


            price = product.find(
                "p",
                class_="price_color"
            ).text.strip()


            star_rating = product.find(
                "p",
                class_="star-rating"
            )["class"][1]


            availability = product.find(
                "p",
                class_="availability"
            ).text.strip()


            book = {
                "title": title,
                "price": price,
                "star_rating": star_rating,
                "availability": availability,
                "category": category
            }


            books_list.append(book)


        # Find next page
        next_button = soup.find(
            "li",
            class_="next"
        )


        # Stop if there is no next page
        if next_button is None:
            break


        # Get next page link
        next_link = next_button.find("a")["href"]


        # Create full next-page URL
        current_url = urljoin(
            current_url,
            next_link
        )


# Convert scraped data into DataFrame
df = pd.DataFrame(books_list)


# Save raw data
df.to_csv(
    "data_pipeline/raw_books.csv",
    index=False
)


# Print final information
print("\nScraping completed.")

print("Total books:", len(df))

print("\nBooks by category:")

print(
    df["category"].value_counts()
)

print("\nRaw data saved to:")
print("data_pipeline/raw_books.csv")