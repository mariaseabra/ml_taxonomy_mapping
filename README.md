# ml_taxonomy_mapping
Retailers Products Classification Model

This repository storages developments from a Data Operations project where the company is web scrapping products from several different retailers, such as Design Within Reach and Discount School Supply. Every time that a new retailer website is scrapped, a dataset is generated. The project's goal is to map the retailers' categorisation to the company's, which is contained Taxonomy.csv.

Mapping process: I check the src_pt, src_cat, and src_sc columns from Retailer dataset and insert in columns ent_pt_2, ent_pt_2, and ent_sc_2 of the same dataset the categories from productType, category, and subCategory columns from Taxonomy.csv that best match the retailer's description.

The developments here are meant to automate the above process with a machine learning algorithm in Python that maps the new datasets' products based on the similarity of the words from already-mapped datasets.
