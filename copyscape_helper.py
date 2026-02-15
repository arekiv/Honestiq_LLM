import sys
import urllib

# Python 2/3 compatibility
isPython2 = sys.hexversion < 0x03000000

if sys.hexversion < 0x02050000:
    import elementtree.ElementTree as CopyscapeTree
else:
    import xml.etree.ElementTree as CopyscapeTree

if isPython2:
    import urllib2
else:
    import urllib.request
    import urllib.error
    import urllib.parse

# Constants
COPYSCAPE_USERNAME = "honestiq"
COPYSCAPE_API_KEY = "74annu5vx0w3e0hz"
COPYSCAPE_API_URL = "http://www.copyscape.com/api/"

def copyscape_api_call(operation, params={}, postdata=None):
    urlparams = {}
    urlparams['u'] = COPYSCAPE_USERNAME
    urlparams['k'] = COPYSCAPE_API_KEY
    urlparams['o'] = operation
    urlparams.update(params)
    
    uri = COPYSCAPE_API_URL + '?'

    request = None
    if isPython2:
        uri += urllib.urlencode(urlparams)
        if postdata is None:
            request = urllib2.Request(uri)
        else:
            request = urllib2.Request(uri, postdata.encode("UTF-8"))
    else:
        uri += urllib.parse.urlencode(urlparams)
        if postdata is None:
            request = urllib.request.Request(uri)
        else:
            request = urllib.request.Request(uri, postdata.encode("UTF-8"))
    
    try:
        response = None
        if isPython2:
            response = urllib2.urlopen(request)
        else:
            response = urllib.request.urlopen(request)
        res = response.read()
        return CopyscapeTree.fromstring(res)
    except Exception as e:
        print(f"Copyscape API Error: {e}")
        return None

def copyscape_api_text_search_internet(text, encoding="UTF-8", full=0):
    params = {}
    params['e'] = encoding
    params['c'] = str(full)
    # The 'csearch' operation is for content search (plagiarism check)
    return copyscape_api_call('csearch', params, text)

def check_plagiarism_copyscape(text):
    """
    Check plagiarism using Copyscape API and return format compatible with app.py
    """
    if not text:
        return {"error": "No text provided"}

    try:
        # Perform search (full=1 might give more details if available, but 0 is default)
        # Using full=1 to get more comparison details if possible, though cost might remain same per search
        root = copyscape_api_text_search_internet(text, "UTF-8", full=1)
        
        if root is None:
            return {"error": "Failed to connect to Copyscape API"}
            
        if root.find('error') is not None:
            return {"error": root.find('error').text}
            
        # Parse results
        sources = []
        total_words_matched = 0
        
        # Word count of the input text (approximate)
        input_word_count = len(text.split())
        
        # 'querywords' usually indicates the number of words in query
        query_words_node = root.find('querywords')
        if query_words_node is not None:
            try:
                input_word_count = int(query_words_node.text)
            except:
                pass

        results = root.findall('result')
        for result in results:
            url = result.find('url').text if result.find('url') is not None else ""
            title = result.find('title').text if result.find('title') is not None else ""
            
            # Copyscape returns 'minwordsmatched' or similar
            # Using 'minwordsmatched' to estimate percentage or if 'pctmatched' exists
            match_percent = 0
            
            # Some Copyscape responses might have 'pctmatched'
            pct_node = result.find('pctmatched')
            if pct_node is not None:
                match_percent = float(pct_node.text)
            else:
                # Fallback calculation
                min_words = result.find('minwordsmatched')
                if min_words is not None:
                    count = int(min_words.text)
                    if input_word_count > 0:
                        match_percent = (count / input_word_count) * 100
            
            # Cap at 100
            if match_percent > 100:
                match_percent = 100
                
            sources.append({
                "link": url,
                "percent": match_percent,
                "count": int(result.find('minwordsmatched').text) if result.find('minwordsmatched') is not None else 0,
                "title": title
            })
            
        # Calculate overall plagiarism percentage
        # A simple method is taking the max of any single source, 
        # as multiple sources might overlap. 
        # Alternatively, if Copyscape provides an aggregate, use that.
        # Copyscape doesn't typically provide a single "originality score" like some others.
        # We will use the highest match percentage as the plagiarism score.
        
        plagiarism_percentage = 0.0
        if sources:
            plagiarism_percentage = max(s['percent'] for s in sources)
            
        unique_percentage = 100.0 - plagiarism_percentage
        if unique_percentage < 0:
            unique_percentage = 0.0

        # Extract cost
        cost = 0.0
        cost_node = root.find('cost')
        if cost_node is not None:
            try:
                cost = float(cost_node.text)
            except:
                pass
        
        print(f"Copyscape Scan Cost: ${cost}")

        return {
            "plagPercent": plagiarism_percentage,
            "uniquePercent": unique_percentage,
            "sources": sources,
            "details": sources,  # Using sources as details
            "cost": cost
        }

    except Exception as e:
        return {"error": str(e)}
