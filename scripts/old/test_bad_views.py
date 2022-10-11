
import pickle
from cca_zoo.models import CCA

with open('/home/mwang8/projects/def-jbpoline/mwang8/ukbb-cca/scripts/bad_views.pkl', 'rb') as file_views:
    views = pickle.load(file_views)

cca = CCA()
cca.fit(views)
