import pandas as pd

def load_data(sheets_url):
    csv_url = sheets_url.replace("/edit#gid=", "/export?format=csv&gid=")
    return pd.read_csv(csv_url)

metacard_metadata_public_gsheets_url = "https://docs.google.com/spreadsheets/d/1t6mfVOJsSmbwtgvYzyYqT6ofSFoJb6HZBl2ZddZis4g/edit#gid=1481496480"
metacard_microbiome_public_gsheets_url = "https://docs.google.com/spreadsheets/d/13o6uYJ-vX1VIMPO1nkH4WYaNUJe3vdVrw6pv8O76S6g/edit#gid=1074439309"
metacard_serum_public_gsheets_url = "https://docs.google.com/spreadsheets/d/1_Z_U_gJ0li1Iy7wwrQGWoZOcQ6fFc1As21omMuuObXc/edit#gid=1093748047"

# import csv files from current directory - we can change this
# metadata = pd.read_csv('metacard_metadata.csv')
# microbiome = pd.read_csv('metacard_microbiome.csv')
# metabolome = pd.read_csv('metacard_serum.csv')
metadata = load_data(metacard_metadata_public_gsheets_url)
microbiome = load_data(metacard_microbiome_public_gsheets_url)
metabolome = load_data(metacard_serum_public_gsheets_url)

