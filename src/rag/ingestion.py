from neo4j import GraphDatabase
import pandas as pd
 
def load_authors(driver: GraphDatabase.driver):
    df = pd.read_excel("src/data/lib_books.xlsx", sheet_name="authors")
    for index, row in df.iterrows():
        name = str(row.get('name', '')).strip() if 'name' in row else ''
        if (not name):
            continue
        
        driver.execute_query("""
            MERGE (a:Author {name: $name})
            """, name=name)
        

def load_locations(driver: GraphDatabase.driver):
    df = pd.read_excel("src/data/lib_books.xlsx", sheet_name="locations")
    for index, row in df.iterrows():
        code = str(row.get('code', '')).strip() if 'code' in row else ''
        description = str(row.get('description', '')).strip() if 'description' in row else ''
        if (not code) or (not description):
            continue
        
        driver.execute_query("""
            MERGE (l:Location {code: $code, description: $description})
            """, code=code, description=description)


def load_books(driver: GraphDatabase.driver):
    #TODO: Create sheets for categories, and publishers
    unique_categories = set()
    unique_publishers = set()
    
    df = pd.read_excel("src/data/lib_books.xlsx", sheet_name="books")
    for index, row in df.iterrows():
        title = str(row.get('title', '')).strip() if 'title' in row else ''
        subtitle = str(row.get('subtitle', '')).strip() if 'subtitle' in row else ''
        norm_desc = str(row.get('normalized_description', '')).strip() if 'normalized_description' in row else ''
        categories = str(row.get('categories', '')).strip() if 'categories' in row else ''
        authors = str(row.get('authors', '')).strip() if 'authors' in row else ''
        publisher = str(row.get('publisher', '')).strip() if 'publisher' in row else ''
        published_date = str(row.get('published_date', '')).strip() if 'published_date' in row else ''
        isbn = str(row.get('isbn', '')).strip() if 'isbn' in row else ''
        bib_num = str(row.get('bib_num', '')).strip() if 'bib_num' in row else ''
        locations = str(row.get('locations', '')).strip() if 'locations' in row else ''
        if (not title) or (not subtitle) or (not isbn) or (not locations):
            continue
        
        driver.execute_query("""
            MERGE (b:Book {title: $title, subtitle: $subtitle, norm_desc: $norm_desc, categories: $categories, authors: $authors, publisher: $publisher, published_date: $published_date, isbn: $isbn, locations: $locations, bib_num: $bib_num})
            """, title=title, subtitle=subtitle, norm_desc=norm_desc, categories=categories, authors=authors, publisher=publisher, published_date=published_date, isbn=isbn, locations=locations, bib_num=bib_num)
        
        locations = locations.split(",")
        for location in locations:
            location = location.strip()
            if not location:
                continue
            
            driver.execute_query("""
                MATCH (b:Book {title: $title, isbn: $isbn})
                MATCH (l:Location {code: $location})
                MERGE (b)-[:LOCATED_IN]->(l)
                """, title=title.strip(), isbn=isbn.strip(), location=location.strip())
        
        authors = authors.split(",")
        for author in authors:
            author = author.strip()
            if not author:
                continue
            
            driver.execute_query("""
                MATCH (b:Book {title: $title, isbn: $isbn})
                MATCH (a:Author {name: $author})
                MERGE (b)-[:WRITTEN_BY]->(a)
                """, title=title, isbn=isbn, author=author)
            
        categories = categories.split("/")
        for category in categories:
            clean_category = category.strip()
            if clean_category not in unique_categories:
                unique_categories.add(clean_category)
                driver.execute_query("""
                    MERGE (c:Category {name: $category}) WITH c
                    MATCH (b:Book {title: $title, isbn: $isbn})
                    MERGE (b)-[:BELONGS_TO]->(c)
                    """, title=title, isbn=isbn, category=clean_category)
                
        publisher = publisher.strip()
        if publisher not in unique_publishers:
            unique_publishers.add(publisher)
            driver.execute_query("""
                MERGE (p:Publisher {name: $publisher}) WITH p
                MATCH (b:Book {title: $title, isbn: $isbn})
                MERGE (b)-[:PUBLISHED_BY]->(p)
                """, title=title, isbn=isbn, publisher=publisher)

def check_books_exist(driver: GraphDatabase.driver) -> bool:
    result = driver.execute_query("""
        MATCH (b:Book) 
        RETURN count(b) as book_count
        LIMIT 1
    """)
    book_count = result.records[0]["book_count"]
    return book_count > 0
