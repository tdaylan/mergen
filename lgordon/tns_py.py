# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 13:34:13 2021

@author: lindsey gordon

tns_py to create search URLs for the TNS

Functions:
    * SN_page: Given the name of a known SNe, returns information about it stripped from the
                TNS listing 
    * CSV_URL: Given the search criteria, returns the URL to run a search + download the first page of 500 results
    * TNS_get_CSV: Retrieves and saves the CSV file 
"""

import requests
from html.parser import HTMLParser 

class TNSParser(HTMLParser):
    """ Parser to strip relevent information off of the website"""
    info_of_interest = list()
    saving = False
    
    def handle_starttag(self, tag, attrs):
        return

    def handle_endtag(self, tag):
        return
    
    def handle_comment(self, data):
        return 
    
    def handle_data(self, data):
        #clear out all the gunky bits that will show up
        if data.strip() != "" and not data.strip().startswith("@") and not data.strip().startswith("<"):
            #start saving
            #print(data.strip())
            if data.strip() == "RA/DEC (2000)":
                self.saving = True
            #save    
            if self.saving:
                self.info_of_interest.append(data)
            #end saving    
            if data.strip() == "Reporter/s":
                self.saving = False
            
def SN_page(SN_name, printfortesting = False):
    """ Retrieves the webpage for the given SN
    Parses HTML and returns the RA and DEC in hours and in decimal form, 
    the type (if it exists) the redshift (if it exists) and the discovery date and magnitude]
    
    """
    url = "https://www.wis-tns.org/object/" + SN_name 
    parser = TNSParser()
    r = requests.get(url)
    
    
    parser.info_of_interest = list()
    parser.feed(r.text)
    returner = parser.info_of_interest
    
    if printfortesting:
        print(r.status_code) #200 indicates a success
        print(parser.info_of_interest)
    
    RA_DEC_hr = ""
    RA_DEC_decimal = ""
    type_sn = ""
    redshift = ""
    discodate = ""
    discomag = ""
    
    
    
    
    for n in range(len(returner)):
        #
        if returner[n] == "RA/DEC (2000)":
            RA_DEC_hr = returner[n+1]
            RA_DEC_decimal = returner[n+2]
        
        if returner[n].startswith("SN"):
            type_sn = returner[n]
        
        if returner[n] == "Redshift" and returner[n+1] != "Discovery Report":
            redshift = returner[n+1]
        if returner[n] == "Discovery Date":
            discodate = returner[n+1]
        if returner[n] == "Discovery Mag":
            discomag = returner[n+1]
            
    r.close()
    del(returner, parser)
        
    return RA_DEC_hr, RA_DEC_decimal, type_sn, redshift, discodate, discomag
        
    


def CSV_URL(date_start = None, date_end = None, discovered_within = None, 
            discovered_within_units = None, unclassified_at = False, classified_sne = False,
            include_frb = False, name = None, name_like = False, discovery_mag_min = None,
            discovery_mag_max= None, redshift_min = None, redshift_max = None, 
            ra_range_min = None, ra_range_max = None, 
            decl_range_min = None, decl_range_max = None):
    """ Produces a URL for searching TNS
    - date_start and date_end in format: "2020-12-11" 
    - discovered within is "2" and units for it is "days" "months" or "years"
    - coords_units is 'arcsec', 'arcmin', or 'deg'
    
    
    possibly add these later: 
        &ra=
        &decl=
        &radius=
        &coords_unit=arcsec
    
        &frb_repeat=all
        &frb_repeater_of_objid=
        &frb_measured_redshift=0
        &frb_dm_range_min=
        &frb_dm_range_max=
        &frb_rm_range_min=
        &frb_rm_range_max=
        &frb_snr_range_min=
        &frb_snr_range_max=
        &frb_flux_range_min=
        &frb_flux_range_max=
    """
    
    prefix = "https://www.wis-tns.org/search?"
    download_suffix = "&num_page=500&format=csv"
    
    #eventually will need to iterate through pages for max results
    #i = 0
    #page_suffix = "&page=" + str(i)
    
    query = ""
    if date_start is not None and date_end is not None:
        query = query + "&date_start%5Bdate%5D=" + date_start
        query = query + "&date_end%5Bdate%5D=" + date_end
    
    if discovered_within is not None and discovered_within_units is not None:
        query = query + "&discovered_period_value=" + discovered_within
        query = query + "&discovered_period_units=" + discovered_within_units
    
    if unclassified_at: #include them
        query = query + "&unclassified_at=1"
            
    if classified_sne: #only classified supernovae
        query = query + "&classified_sne=1"
    
    if include_frb: #include frbs in results
        query = query + "&include_frb=1"
    
    if name is not None: #if searching by name
        query = query + "&name" + name
        if name_like:
            query = query + "&name_like=1"
    
    if discovery_mag_min is not None:
        query = query + "&discovery_mag_min=" + discovery_mag_min
    
    if discovery_mag_max is not None:
        query = query + "&discovery_mag_max=" + discovery_mag_max
    
    if redshift_min is not None:
        query = query + "&redshift_min=" + redshift_min
        
    if redshift_max is not None:
        query = query + "&redshift_max=" + redshift_max
        
    if ra_range_min is not None:
         query = query + "&ra_range_min=" + ra_range_min
     
    if ra_range_max is not None:
         query = query + "&ra_range_max=" + ra_range_max
    
    if decl_range_min is not None:
        query = query + "&decl_range_min=" + decl_range_min
    
    if decl_range_max is not None:
        query = query + "&decl_range_max=" + decl_range_max
    
    url = prefix + query + download_suffix #+ page_suffix
    
    return url


    
def TNS_get_CSV(savepath, filelabel, url):
    """ Function to download a CSV from TNS given the URL for that search"""

    for i in range(2): #ifthere's more than 1000 results...sucks
        savefile = savepath + filelabel + "-" + str(i) + ".csv"
        url = url + "&page=" + str(i)
        data = requests.get(url, allow_redirects = True)
        #print(data.content)
        #print("\n\n\n")
        #ie,"./filelabel-1.png"
        with open(savefile, 'wb') as f:
            f.write(data.content)

 #%% 
#TESTING
#RA_DEC_hr, RA_DEC_decimal, type_sn, redshift, discodate, discomag = SN_page("2019dke", printfortesting = True)
#SN_page("2018eod", printfortesting=True)
#print(RA_DEC_hr, RA_DEC_decimal, type_sn, redshift, discodate, discomag)  
#url = CSV_URL(date_start = "2020-11-10", date_end = "2020-11-12", discovered_within = None, 
 #           discovered_within_units = None, unclassified_at = False, classified_sne = False,
  #          include_frb = False, name = None, name_like = False, discovery_mag_min = None,
   #         discovery_mag_max= None)
#print(url)    
#savehere = "C:/Users/conta/Downloads/"
#filelabel= "testinggetcsv"
#TNS_get_CSV(savehere,filelabel, url)  

    
    
    
    
    
    
    
    
    
    
    
    
    
    