#Preprocessing related Imports
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelBinarizer, OneHotEncoder, FunctionTransformer
from sklearn.base import BaseEstimator,TransformerMixin


# Numerical Features
#skewed_num_attrib = ['SQUARE_FT']
#other_num_attribs=['BHK_NO.', 'LONGITUDE','LATITUDE']
# Categorical Features
#label_cat_attribs=[ 'UNDER_CONSTRUCTION', 'RERA', 'RESALE'] # 3?
#onehot_cat_attribs = ['POSTED_BY', 'BHK_OR_RK'] # 5
#city_attrib = ['ADDRESS'] # 3
#bhk_no_attrib = ['BHK_NO.']

#column_names = skewed_num_attrib + other_num_attribs + label_cat_attribs + onehot_cat_attribs + city_attrib # + ['bnk_lt_8']
#column_names_encoded = skewed_num_attrib + other_num_attribs + label_cat_attribs + [ 'POSTED_BY_dealer', 'POSTED_BY_owner',  'BHK_OR_RK_rk', 'ADDRESS_tier2', 'ADDRESS_tier3'] # + ['bnk_lt_8']
# Help Functions

tier1 = {'Ahmedabad', 'Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai', 'Pune', 'Maharashtra'}
tier2 = {'Agra', 'Ajmer', 'Aligarh', 'Amravati', 'Amritsar', 'Asansol', 'Aurangabad', 'Bareilly', 
                  'Belgaum', 'Bhavnagar', 'Bhiwandi', 'Bhopal', 'Bhubaneswar', 'Bikaner', 'Bilaspur', 'Bokaro Steel City', 
                  'Chandigarh', 'Coimbatore', 'Cuttack', 'Dehradun', 'Dhanbad', 'Bhilai', 'Durgapur', 'Dindigul', 'Erode', 
                  'Faridabad', 'Firozabad', 'Ghaziabad', 'Gorakhpur', 'Gulbarga', 'Guntur', 'Gwalior', 'Gurgaon', 'Guwahati', 
                  'Hamirpur', 'Hubli–Dharwad', 'Indore', 'Jabalpur', 'Jaipur', 'Jalandhar', 'Jammu', 'Jamnagar', 'Jamshedpur', 
                  'Jhansi', 'Jodhpur', 'Kakinada', 'Kannur', 'Kanpur', 'Karnal', 'Kochi', 'Kolhapur', 'Kollam', 'Kozhikode', 
                  'Kurnool', 'Ludhiana', 'Lucknow', 'Madurai', 'Malappuram', 'Mathura', 'Mangalore', 'Meerut', 'Moradabad', 
                  'Mysore', 'Nagpur', 'Nanded', 'Nashik', 'Nellore', 'Noida', 'Patna', 'Pondicherry', 'Purulia', 'Prayagraj', 
                  'Raipur', 'Rajkot', 'Rajahmundry', 'Ranchi', 'Rourkela', 'Ratlam', 'Salem', 'Sangli', 'Shimla', 'Siliguri', 
                  'Solapur', 'Srinagar', 'Surat', 'Thanjavur', 'Thiruvananthapuram', 'Thrissur', 'Tiruchirappalli', 'Tirunelveli', 
                  'Tiruvannamalai', 'Ujjain', 'Bijapur', 'Vadodara', 'Varanasi', 'Vasai-Virar City', 'Vijayawada', 'Visakhapatnam', 
                  'Vellore', 'Warangal'}
def city2tier(city):
    if city in tier1:
        return 'tier1'
    elif city in tier2:
        return 'tier2'
    else:
        return 'tier3'    
    
def addr2tier(addr_arr):
    a = map(lambda row: [city2tier(row[0].split(',')[-1])], addr_arr)
    return list(a)

def bnk_lt_8(bnk_arr):
    a = map(lambda row: [1 if row[0] < 8 else 0], bnk_arr)
    return list(a)
                         
# Transformers
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

#log_transformer = 
#addr2tier_transformer = 

class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False):
        self.sparse_output = sparse_output
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        enc = LabelBinarizer(sparse_output=self.sparse_output)
        return enc.fit_transform(X)