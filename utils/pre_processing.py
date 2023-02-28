import re

def remove_html_tags(text):
    """Remove html tags from a string"""
    text=text.replace('&nbsp;','')
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def removeParenthesis(line):
    #remove a) b) [12]...
    #remove a) in the middle of sentences
    line = re.sub(r"\ [0-9a-zA-Z]*?\)|\]", '', line)
    #remove a) in the beginning of sentences
    return re.sub(r"^\"?[0-9a-zA-Z]*?\)|\]", '', line)

def replaceRoman(text):
    #remove Roman numbers
    text = re.sub(r'(\b[mdclcvi]+\b)(\.)\ ?', '', text)
    text = re.sub(r'(\b[MDCLXVI]+\b)(\.)\ ?', '', text)
    text = re.sub(r'(\b[mdclcvi]+\b)\ ?', '', text)
    text = re.sub(r'(\b[MDCLXVI]+\b)\ ?', '', text)
    text = re.sub(r'\([mdclcvi]+\)', '', text)
    text = re.sub(r'\([MDCLXVI]+\)', '', text)
    return text

def removeWeirdChars(text):
    #remove uncommon chars
    text = text.replace(chr(176), "")
    text = text.replace("º", "")
    text = text.replace("ª", "")
    text = text.replace(chr(8226), "")
    text = text.replace(chr(8220), '"')
    text = text.replace("[", "")
    text = text.replace("]", "")
    text = text.replace("*", "")
    #text = text.replace("(...)", "")
    text = text.replace(chr(40), '')
    text = text.replace(chr(8230), '')
    text = text.replace('“', '')
    text = text.replace('”', '')
    return text

def removeSectionsNumber(text):
    #remove initial sections numbers 3.1-
    text = re.sub(r"^\;?([0-9]*(\.)?)*(-)?", '', text)
    text = re.sub(r"[0-9](\.)", '', text)
    text = re.sub(r"^[A-Z]\ -", '', text)
    return text

def semicolons(text):
    #remove semicolons from the begining
    return re.sub(r"^(;)*", '', text)

def cleanText(text):
    #join all the functions
    #????
    return [semicolons(removeSectionsNumber(removeParenthesis(removeWeirdChars(line)))) for line in re.split('\n', remove_html_tags(replaceRoman(text)))]

def selectSentencesWithAtLeastNWords(cleanText, NWords):
    #return all the entries with at least n words
    return [sentence for sentence in cleanText if len(sentence.split(" "))>NWords]

def removeUnicode(text):
    return re.sub(r"\xa0*", '', text)

def is_mainly_text(text):
    """function that receives a string and sees if 80% follows the pattern [a-zA-Z]"""
    original_len = len(text)

    amount_of_spaces = text.count(' ')
    if amount_of_spaces/original_len > 0.5:
        return False

        
    text_only_letters = re.sub(r"[a-zA-Z]", '', text)
    final_len = len(text_only_letters)

    percentage = 1-(final_len/original_len)

    print(percentage)
    if percentage > 0.8:
        return True
