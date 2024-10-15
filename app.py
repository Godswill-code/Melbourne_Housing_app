import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="Melbourne!!!", page_icon=":bar_chart:", layout="wide")

st.title(":bar_chart: Melbourne Analytics App")
st.markdown('<style>div.block-container{padding-top:2rem;}<style>', unsafe_allow_html=True)

df = pd.read_csv("melbourne.csv")


#st.sidebar.image("data/mel.png", caption="Melbourne suburbs Analytics")

#Data cleaning
#Missing value(mean imputation for numerical columns)
df.isnull().sum()
mean_value = df[['BuildingArea', 'Price', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'YearBuilt', 'Longtitude', 'Lattitude', 'Propertycount']].mean()
df_mean = df.fillna(mean_value)
df_mean.isnull().sum()

#Mode imputation for categorical columns
# Calculate the mode for the specified columns.
mode_values = df_mean[['CouncilArea', 'Regionname']].mode()

# Check if mode_values is not empty
if not mode_values.empty:
    # Use .iloc[0] to access the first row (which contains the mode values)
    # and fill null values only if mode_values is not empty to avoid the error
    df_mean[['CouncilArea', 'Regionname']] = df_mean[['CouncilArea', 'Regionname']].fillna(mode_values.iloc[0])
df_mean.isnull().sum()

#Duplicates
duplicates = df_mean.duplicated()
duplicate_rows = df_mean[df_mean.duplicated()]
print(duplicate_rows)
df_dup = df_mean.drop_duplicates()

#Duplicates
df_dup['log_price'] = np.log(df_dup['Price'] + 1)
df_dup['log_price'].hist(bins=20)
df = df_dup
df

region = df['Regionname'].unique()
st.sidebar.selectbox('Choose a region', region)

st.sidebar.selectbox('House Type', ['h','t','u'])

st.bar_chart(df, y='Price', x='Distance')





#The dependent (or target) variable we are trying to predict in this analysis is Price. It appears normally distributed but skewed to the right.
plt.figure(figsize=(16,7))
p_chart = sns.histplot(df['Price'], kde=False, edgecolor="k")




house_type = df['Type'].value_counts().to_frame()
x = house_type.index
y = house_type['count']

house_typefig = px.bar(x=x,y=y)

st.plotly_chart(house_typefig)
st.markdown('The house type h has the highest number of count')

st.markdown('Count of different number of rooms in Melbourne')
Rooms_dist = df['Rooms'].value_counts()
st.markdown(
    "Rooms distribution"
)
st.bar_chart(Rooms_dist)

df.groupby('Rooms')['Price'].mean().to_frame()
fig = px.box(df, x='Rooms',y='Price', title='Price Distribution')
st.plotly_chart(fig)

fig = px.box(df, x='Type',y='Distance', title='Distance Distribution')
st.plotly_chart(fig)

#convert the 'Date' column in the data to a datetime object
df['Date']= pd.to_datetime(df['Date'],dayfirst=True)
fig, ax = plt.subplots()
ax.set_xlim([pd.Timestamp('2016-01-01'), pd.Timestamp('2018-01-01')])

# Grouping the features by Date and selecting only numeric columns
std_c = df[df['Type']=='h'].sort_values('Date',ascending=False).groupby('Date').agg({'Rooms':'std', 'Price':'std', 'Distance':'std','Postcode':'std',
                                                                                         'Bedroom2':'std', 'Bathroom':'std', 'Car':'std', 'Landsize':'std',
                                                                                         'BuildingArea':'std', 'YearBuilt':'std', 'Lattitude':'std', 'Longtitude':'std', 'Propertycount':'std'})
count = df[df['Type']=='h'].sort_values('Date',ascending=False).groupby('Date').agg({'Rooms':'count', 'Price':'count', 'Distance':'count','Postcode':'count',
                                                                                         'Bedroom2':'count', 'Bathroom':'count', 'Car':'count', 'Landsize':'count',
                                                                                         'BuildingArea':'count', 'YearBuilt':'count', 'Lattitude':'count', 'Longtitude':'count', 'Propertycount':'count'})
mean = df[df['Type']=='h'].sort_values('Date',ascending=False).groupby('Date').agg({'Rooms':'mean', 'Price':'mean', 'Distance':'mean','Postcode':'mean',
                                                                                         'Bedroom2':'mean', 'Bathroom':'mean', 'Car':'mean', 'Landsize':'mean',
                                                                                         'BuildingArea':'mean', 'YearBuilt':'mean', 'Lattitude':'mean', 'Longtitude':'mean', 'Propertycount':'mean'})

std_c.head(15)
count.head(15)
mean.head(15)

mean["Price"].plot(yerr=std_c["Price"],ylim=(400000,1500000))
st.pyplot(fig)


st.write(
"I decided to check how all the variables are correlated with one another. The dataset includes the home's construction year which is the variable 'YearBuilt'. This data point actually reflects the home's age. To simplify analysis and visualization, we can categorize the home's age into two groups:Historic( older homes, built over 50 years ago), and Contemporary (newer homes, built within the last 50 years)")
# Add age variable
df['HomeAge'] = 2017 - df['YearBuilt']

# Identify historic homes
df['Historic'] = np.where(df['HomeAge']>=50,'Historic','Contemporary')

# Convert to Category
df['Historic'] = df['Historic'].astype('category')

# Select only numerical features for correlation analysis
numerical_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(10,6))
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(numerical_df.corr(),cmap = 'coolwarm',linewidth = 1,annot= True, annot_kws={"size": 9})
plt.title('Variable Correlation')
st.pyplot(fig)

