import re
import json
import difflib
import operator
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
with open("cotton_mix.json", "r") as f:
    dataset = json.load(f)

ALL_FIELDS = list(dataset[0].keys())
ALL_PRODUCTS = set(d.get("SAP Material Codes") for d in dataset)
ALL_LOTS = set(d.get("LotNo") for d in dataset)
ops = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "=": operator.eq}

CRITICAL_INSTRUCTIONS = """CRITICAL INSTRUCTIONS:
1. Provide detailed, well-structured responses in 2-5 lines of complete paragraph format
2. Include specific numbers, values, and lot details from the dataset to support your recommendations
3. Explain the reasoning behind your suggestions with data-backed evidence
4. Structure your response with: a) Analysis of current data, b) Specific recommendations with ranges, c) Expected outcomes/benefits
5. Use professional but easy-to-understand language with concrete examples
6. If unrelated to cotton/textile, reply ONLY: "Sorry! please ask query's related to Cotton Mix."
7. Never hallucinate - only use actual data from the dataset"""

def entry_to_sentence(entry):
    return ". ".join([f"{k} is {v}" for k, v in entry.items()])

def create_rag_documents():
    documents = []
    for idx, record in enumerate(dataset):
        content = f"""Cotton Lot {record.get('LotNo', 'Unknown')} (Material: {record.get('SAP Material Codes', 'Unknown')}):
- Product: {record.get('Product Text', 'N/A')}, Location: {record.get('Sloc', 'N/A')}, Station: {record.get('Station code', 'N/A')}
- Quantity: {record.get('Qty -BALES', 'N/A')} bales, Weight: {record.get('KG/BALE', 'N/A')}, Total: {record.get('KGS', 'N/A')} KG, Value: {record.get('Value', 'N/A')}
- Trash: {record.get('TRASH%', 'N/A')}%, Moisture: {record.get('MOISTURE', 'N/A')}%, MIC: {record.get('MIC', 'N/A')}, SCI: {record.get('SCI', 'N/A')}
- MAT: {record.get('MAT', 'N/A')}, LEN: {record.get('LEN', 'N/A')}, UNF: {record.get('UNF', 'N/A')}, SFI: {record.get('SFI', 'N/A')}"""
        
        metadata = {k: record.get(k, 0 if k in ['Qty -BALES', 'KGS', 'Value', 'TRASH%', 'MOISTURE', 'MIC', 'SCI', 'MAT', 'LEN', 'UNF', 'SFI'] else "") 
                   for k in ['SAP Material Codes', 'LotNo', 'Sloc', 'Station code', 'Qty -BALES', 'Value', 'TRASH%', 'MOISTURE', 'MIC']}
        metadata['idx'] = idx
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

print("Building RAG system for Cotton Mix...")
rag_documents = create_rag_documents()
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                        model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
vector_db = FAISS.from_documents(rag_documents, embedding_model)
print(f"RAG system ready with {len(rag_documents)} cotton lot documents indexed.")


def detect_number(query):
    nums = re.findall(r"[+\-]?\d+\.?\d*", query)
    return float(nums[0]) if nums else None

def detect_field(query):
    q = query.lower().strip()
    for f in ALL_FIELDS:
        if f.lower() in q:
            return f
    words = [w for w in q.split() if len(w) > 2]
    for w in words:
        close = difflib.get_close_matches(w, [f.lower() for f in ALL_FIELDS], cutoff=0.6)
        if close:
            return next((f for f in ALL_FIELDS if f.lower() == close[0]), None)
    return next((f for f in ALL_FIELDS if any(w in f.lower() for w in words)), None)

def reasoning_mode(field, value):
    vals = [float(d.get(field)) for d in dataset if isinstance(d.get(field), (int, float))]
    if not vals:
        return None
    mn, mx = min(vals), max(vals)
    if value > mx:
        return f"{field} = {value} is ABOVE max {mx}. Unusual high value."
    if value < mn:
        return f"{field} = {value} is BELOW min {mn}. Unusual low value."
    return f"{field} = {value} within dataset range ({mn}-{mx})."

def numeric_filter(query):
    results = []
    for match in re.finditer(r"([a-zA-Z %\[\]]+)\s*(>=|<=|>|<|=)\s*([+\-]?\d+\.?\d*)", query):
        field_txt, op_str, val = match.groups()
        field_key = detect_field(field_txt)
        if field_key:
            threshold = float(val)
            op_func = ops[op_str]
            results.extend([entry_to_sentence(d) for d in dataset 
                          if isinstance(d.get(field_key), (int, float)) and op_func(d.get(field_key), threshold)])
    return results if results else None

def get_contextual_documents(query, k=15):
    q_lower = query.lower()
    if any(word in q_lower for word in ['compare', 'difference', 'pattern', 'correlation']):
        return sorted(rag_documents, key=lambda d: d.metadata.get('Value', 0), reverse=True)[:k//2] + \
               sorted(rag_documents, key=lambda d: d.metadata.get('Value', 0))[:k//2]
    if 'trash' in q_lower:
        return sorted(rag_documents, key=lambda d: d.metadata.get('TRASH%', 0), reverse=True)[:k]
    if 'moisture' in q_lower:
        return sorted(rag_documents, key=lambda d: d.metadata.get('MOISTURE', 0), reverse=True)[:k]
    return vector_db.similarity_search(query, k=k)


def get_statistics(field_name):
    values = [d.get(field_name) for d in dataset if isinstance(d.get(field_name), (int, float))]
    if not values:
        return None
    return {'min': min(values), 'max': max(values), 'avg': sum(values) / len(values), 'count': len(values)}


def get_statistics_by_quality(field_name):
    all_values_data = [(d.get(field_name), d.get('Value', 0)) for d in dataset 
                       if isinstance(d.get(field_name), (int, float)) and isinstance(d.get('Value'), (int, float))]
    if not all_values_data:
        return None
    median_value = sorted([v[1] for v in all_values_data])[len(all_values_data)//2]
    high_quality = [v[0] for v in all_values_data if v[1] >= median_value]
    low_quality = [v[0] for v in all_values_data if v[1] < median_value]
    result = {}
    if high_quality:
        result['high_value'] = {'min': min(high_quality), 'max': max(high_quality), 
                               'avg': sum(high_quality) / len(high_quality), 'count': len(high_quality)}
    if low_quality:
        result['low_value'] = {'min': min(low_quality), 'max': max(low_quality), 
                              'avg': sum(low_quality) / len(low_quality), 'count': len(low_quality)}
    return result


def detect_intent_llm(query):
    try:
        intent_prompt = f"""Classify the user's intent into ONE category only:
Categories: recommendation, alert, prediction, statistics, explanation, listing, general
User Query: "{query}"
Respond with ONLY ONE WORD from the categories above."""
        
        response = client.chat.completions.create(model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": intent_prompt}], temperature=0.1, max_tokens=10)
        intent = response.choices[0].message.content.strip().lower()
        valid_intents = ['recommendation', 'alert', 'prediction', 'statistics', 'explanation', 'listing', 'general']
        return next((v for v in valid_intents if v in intent), 'general')
    except Exception as e:
        print(f"Intent detection error: {e}")
        return 'general'


def assess_quality_level(query, field=None, value=None):
    q_lower = query.lower()
    quality_indicators = {'poor': ['poor', 'low quality', 'bad', 'inferior', 'substandard'],
        'moderate': ['moderate', 'average', 'medium', 'acceptable'],
        'good': ['good', 'high quality', 'premium', 'excellent', 'superior']}
    
    for level, keywords in quality_indicators.items():
        if any(kw in q_lower for kw in keywords):
            return level.upper()
    
    if field and value is not None:
        stats = get_statistics(field)
        if stats:
            avg = stats['avg']
            if 'TRASH' in field.upper() or 'MOISTURE' in field.upper():
                return 'GOOD' if value <= avg * 0.8 else ('MODERATE' if value <= avg * 1.2 else 'POOR')
            else:
                return 'GOOD' if value >= avg * 1.2 else ('MODERATE' if value >= avg * 0.8 else 'POOR')
    return 'MODERATE'


def generate_llm_response(query, context_docs, intent_type):
    field, value = detect_field(query), detect_number(query)
    prompts = {
        'alert': lambda: f"""You are a cotton quality analyst. Quality Level: {assess_quality_level(query, field, value)}
Dataset: {chr(10).join(context_docs[:15])}
Query: {query}
Provide detailed alert in paragraph format explaining: 1) Quality assessment with specific values, 2) Concerning parameters and why they matter, 3) Recommended actions

{CRITICAL_INSTRUCTIONS}""",
        
        'recommendation': lambda: f"""You are a cotton procurement expert.
Dataset: {chr(10).join([f"- {f}: {get_statistics_by_quality(f)}" for f in ['TRASH%', 'MOISTURE', 'MIC'] if get_statistics_by_quality(f)])}
{chr(10).join(context_docs[:10])}
Query: {query}
Provide detailed recommendation in paragraph format explaining: 1) Why these specific lots/products are recommended, 2) The quality parameters and their significance, 3) Expected value and benefits

{CRITICAL_INSTRUCTIONS}""",
        
        'prediction': lambda: f"""You are a cotton analyst.
Patterns: Total lots: {len(dataset)}, Avg value: {sum(d.get('Value',0) for d in dataset)/len(dataset):.2f}
{chr(10).join(context_docs[:10])}
Query: {query}
Provide detailed prediction in paragraph format explaining: 1) Current patterns observed in the data, 2) Probabilistic forecast with reasoning, 3) Confidence level and factors affecting prediction

{CRITICAL_INSTRUCTIONS}""",
        
        'explanation': lambda: f"""You are a cotton quality expert.
Patterns: High trash (>14%): {sum(1 for d in dataset if d.get('TRASH%',0)>14)}/{len(dataset)} lots
{chr(10).join(context_docs[:10])}
Query: {query}
Provide detailed explanation in paragraph format covering: 1) The concept and its importance, 2) How it relates to the dataset with specific correlations, 3) Practical implications

{CRITICAL_INSTRUCTIONS}"""
    }
    
    try:
        prompt = prompts.get(intent_type, lambda: f"Context: {chr(10).join(context_docs[:10])}\nQuery: {query}\nAnswer briefly:")()
        response = client.chat.completions.create(model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=400)
        return response.choices[0].message.content
    except:
        return f"Analysis: {chr(10).join(context_docs[:500])}"


def handle_listing_queries(query):
    q_lower = query.lower()
    if any(kw in q_lower for kw in ['list', 'show', 'which lots', 'find', 'count', 'how many']):
        is_high = any(w in q_lower for w in ['high quality', 'premium', 'best', 'excellent'])
        is_low = any(w in q_lower for w in ['low quality', 'poor', 'issues', 'concerns'])
        if is_high or is_low:
            lots = sorted(dataset, key=lambda d: d.get('Value', 0), reverse=is_high)[:10]
            result = f"Found {len(lots)} {'highest' if is_high else 'lowest'} value cotton lots:\n"
            for lot in lots:
                result += f"\n{lot.get('LotNo')} - {lot.get('Product Text')} (Value: {lot.get('Value')}, Station: {lot.get('Station code')})"
            return result
    return None


def handle_statistics_queries(query):
    if any(kw in query.lower() for kw in ['average', 'avg', 'mean', 'maximum', 'max', 'minimum', 'min', 'statistics', 'stats', 'range']):
        field = detect_field(query)
        if field:
            stats = get_statistics_by_quality(field)
            if stats:
                result = f"{field} Statistics:\n"
                for key in ['high_value', 'low_value']:
                    if key in stats:
                        result += f"{key.replace('_', '-').title()}: avg={stats[key]['avg']:.2f}, range={stats[key]['min']}-{stats[key]['max']}\n"
                return result
    return None


def answer_query(query):
    intent = detect_intent_llm(query)
    print(f"[Intent Detected: {intent}]")
    
    # Quick handlers for statistics and listing
    if intent == 'statistics':
        result = handle_statistics_queries(query)
        if result:
            return result
    if intent == 'listing':
        result = handle_listing_queries(query)
        if result:
            return result
    
    # Gather context documents
    context_docs = []
    field, number = detect_field(query), detect_number(query)
    
    # Handle specialized intents with LLM generation
    if intent in ['alert', 'recommendation', 'prediction', 'explanation']:
        retrieved_docs = get_contextual_documents(query, k=15)
        context_docs.extend([d.page_content for d in retrieved_docs])
        return generate_llm_response(query, context_docs, intent)
    
    # Reasoning mode for field+value queries
    if number is not None and field is not None:
        r = reasoning_mode(field, number)
        if r and intent in ['alert', 'recommendation']:
            retrieved_docs = get_contextual_documents(query, k=10)
            return generate_llm_response(query, [r] + [d.page_content for d in retrieved_docs], intent)
        if r:
            return r
    
    # Numeric filtering and extraction
    num_res = numeric_filter(query)
    if num_res:
        context_docs.extend(num_res)
    
    for prod in [p for p in ALL_PRODUCTS if p and p.lower() in query.lower()]:
        context_docs.extend([entry_to_sentence(d) for d in dataset if d.get("SAP Material Codes") == prod])
    
    for lot in [l for l in ALL_LOTS if l and l.lower() in query.lower()]:
        context_docs.extend([entry_to_sentence(d) for d in dataset if d.get("LotNo") == lot])
    
    # RAG retrieval if no context
    if not context_docs:
        context_docs = [d.page_content for d in get_contextual_documents(query, k=20)]
    
    # Build summary and generate response
    total_lots = len(dataset)
    total_bales = sum(d.get("Qty -BALES", 0) for d in dataset if isinstance(d.get("Qty -BALES"), (int, float)))
    total_value = sum(d.get("Value", 0) for d in dataset if isinstance(d.get("Value"), (int, float)))
    
    context_text = f"""Dataset: {total_lots} lots, {total_bales} bales, value: {total_value}
Fields: {', '.join(ALL_FIELDS)}

Relevant Data:
{chr(10).join(context_docs[:25])}"""
    
    try:
        prompt = f"""You are a cotton quality analyst. Intent: {intent}
{context_text}
Query: {query}

{CRITICAL_INSTRUCTIONS}

Answer:"""
        
        response = client.chat.completions.create(model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=400)
        return response.choices[0].message.content
    except Exception as e:
        return f"Analysis: {context_text[:1500]}" if context_docs else f"Error: {str(e)}"


if __name__ == "__main__":
    print("\n=== Cotton Mix Bot Ready ===\n")
    test_queries = [
        "What is the average trash percentage?",
        "Show me high quality cotton lots",
        "Which lots have moisture above 7?",
        "Compare CAH and CMH products",
        "What factors affect cotton value?"
    ]
    print("Example queries:")
    for q in test_queries:
        print(f"  - {q}")
    print("\n" + "="*50 + "\n")
