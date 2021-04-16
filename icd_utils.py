import re
p_icd10 = re.compile("[A-TV-Z][0-9][0-9AB]\.?[0-9A-TV-Z]{0,4}") # re.IGNORECASE

def encode_(code, sep='.', precision=None): 
    """
    Input
      code: a string of ICD10 code 
    
    Output
      A string of ICD10 code with the dot removed while retaining only n subcategory digits, where 
      n is specified via precision
    """
    if code.find(sep) < 0: return code # no-op
    base, *subcat = str(code).split(sep)
    if len(subcat) > 0: 
        # assert len(subcat) == 1  # ... ok
        if precision is not None: 
            return ''.join((base, subcat[0][:precision]))
        else: 
            return ''.join((base, subcat[0]))
    return str(base)
def encode(codes, sep='.', precision=None):
    if isinstance(codes, (str, int)): 
        return encode_(codes, sep, precision)
    # assert isinstance(code, (list, set, np.ndarray))
    return [encode(code, sep, precision) for code in codes]

def decode_(code, sep='.', precision=None):
    if code.find(sep) > 0: 
        return code  # no-op
    base, subcat = code[:3], code[3:]
    # print(f"base: {base}, subcat: {subcat}")
    if len(subcat) > 0:
        if precision is not None: 
            return sep.join((base, subcat[:precision]))
        else: 
            return sep.join((base, subcat))
    return base
def decode(codes, sep='.', precision=None):
    if isinstance(codes, (str, int)): 
        return decode_(codes, sep, precision)
    # assert isinstance(code, (list, set, np.ndarray))
    return [decode(code, sep, precision) for code in codes]

def is_valid_icd(code, ignore_case=False):
    if ignore_case: 
        m = p_icd10.match(code, re.IGNORECASE)
    else: 
        m = p_icd10.match(code) 
    return True if p_icd10.match(code) is not None else False    

def is_disease_specific(code):
    """
    Returns True if the ICD10 code is a disease-specific diagnosis. 
    
    e.g.
        A00–B99 is disease-specific because they represent 
        'certain infectious and parasitic diseases' 
        
        whereas 
        
        O00–O99 representing 'pregnancy, childbirth and the puerperium' 
        is not
    """
    return True if is_valid_icd(code) and code < 'O00' else False

def test(): 
    pass

if __name__ == "__main__":
    test()