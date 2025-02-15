import spacy
from spacy.language import Language

# Load a blank spaCy model
nlp = spacy.blank("en")

# Define and register the custom NER component
@Language.component("custom_ner")
def custom_ner(doc):
    """
    Simulates NER for patient data and medications.
    """
    entities = [
        ("John Doe", "PERSON"),
        ("45 years", "AGE"),
        ("Paracetamol", "DRUG"),
        ("500mg", "DOSE"),
        ("orally", "ROUTE"),
        ("twice daily", "FREQ"),
    ]
    spans = [doc.char_span(doc.text.find(ent[0]), doc.text.find(ent[0]) + len(ent[0]), label=ent[1]) for ent in entities if ent[0] in doc.text]
    doc.ents = [span for span in spans if span]
    return doc

# Add the custom NER component to the pipeline
nlp.add_pipe("custom_ner")

# Standardization dictionary
standardization_dict = {
    "John Doe": "Patient 1",
    "45 years": "45",
    "Paracetamol": "Acetaminophen",
    "orally": "oral",
    "twice daily": "2x per day",
}

# Main pipeline
def process_text(text):
    """
    Preprocess, extract entities, and standardize them.
    """
    # Preprocess text
    text = text.strip().lower()

    # Extract entities
    doc = nlp(text)
    entities = {ent.text: ent.label_ for ent in doc.ents}

    # Standardize entities
    standardized = {standardization_dict.get(ent, ent): label for ent, label in entities.items()}

    # Print results
    print(f"Extracted Entities: {entities}")
    print(f"Standardized Entities: {standardized}")

# Test the pipeline
if __name__ == "__main__":
    test_text = "John Doe, a 45 years old patient, was prescribed Paracetamol 500mg to be taken orally twice daily."
    process_text(test_text)




