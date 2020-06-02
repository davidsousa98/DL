import pandas as pd
import zipfile as zp

def get_files_zip():
    with zp.ZipFile("./data/Game_stats.zip") as myzip:
        list = myzip.namelist()
    return list


def read_csv_zip(files):
    df_list = []
    for file in files:
        with zp.ZipFile("./data/Game_stats.zip") as myzip:
            with myzip.open(file) as myfile:
                df_list.append(pd.read_csv(myfile))
    return df_list


# Reading files from zip without extracting them
# Warning is returned however it's not concerning!
get_files_zip()
calend_df, host_df, list_df, neighb_df, prop_df = read_csv_zip(['lisbon_0619_dailycalendar.csv',
                                                                         'lisbon_0619_hosts.csv',
                                                                         'lisbon_0619_listings.csv',
                                                                         'lisbon_0619_neighbourhoods.csv',
                                                                         'lisbon_0619_property.csv'])

df1 = list_df.loc[:, ["listing_id", "host_id", "property_id", "room_type", "amenities", "square_feet", "price",
                      "cleaning_fee", "availability_next_30", "availability_next_60", "availability_next_90",
                      "reviews_per_month", "review_scores_rating", "review_scores_accuracy",
                      "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication",
                      "review_scores_location", "review_scores_value", "cancellation_policy", "listing_url"]]

df3 = host_df.loc[:, ["host_id", "host_since", "host_response_time", "host_response_rate", "host_is_superhost",
                      "host_listings_count", "host_has_profile_pic", "host_identity_verified"]]
df3.drop_duplicates(inplace=True)

df4 = prop_df.loc[:, ["property_id", "neighbourhood_cleansed", "neighbourhood_group_cleansed", "latitude", "longitude",
                      "property_type"]]

abt_df = df1.merge(df3, how="left", on="host_id").merge(df4, how="left", on="property_id")
