"""
Convert Pitt Corpus CHAT files to JSONL format for fine-tuning
Processes .cha files and creates training dialogues
"""

import json
import re
from pathlib import Path
from typing import List, Dict

def extract_utterances(cha_file: Path) -> List[Dict[str, str]]:
    """Extract participant utterances from a .cha file"""
    utterances = []
    
    with open(cha_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract lines with *PAR: (participant) and *INV: (investigator)
    lines = content.split('\n')
    
    for line in lines:
        if line.startswith('*INV:'):
            # Extract investigator question/prompt
            text = line.split('*INV:')[1].strip()
            # Remove timestamps and markers
            text = re.sub(r'\d+_\d+', '', text)
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'\+\w+', '', text)
            text = text.strip('. \t')
            if text and len(text) > 3:
                utterances.append({"role": "user", "text": text})
        
        elif line.startswith('*PAR:'):
            # Extract participant response
            text = line.split('*PAR:')[1].strip()
            # Remove timestamps, markers, and hesitations
            text = re.sub(r'\d+_\d+', '', text)
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'\+\w+', '', text)
            text = re.sub(r'&-\w+', '', text)  # Remove &-um, &-uh
            text = re.sub(r'<.*?>', '', text)  # Remove <text>
            text = re.sub(r'\[\/\]', '', text)
            text = re.sub(r'\[\/\/\]', '', text)
            text = text.strip('. \t')
            if text and len(text) > 3:
                utterances.append({"role": "assistant", "text": text})
    
    return utterances


def create_training_sample(utterances: List[Dict], is_dementia: bool) -> Dict:
    """Create a training sample from utterances"""
    
    if not utterances:
        return None
    
    # System prompt
    system_prompt = """You are a caring and patient assistant designed to support elderly individuals, especially those with dementia or memory concerns.

Your communication style:
- Use simple, clear language
- Speak slowly and warmly
- Be patient and empathetic
- Avoid complex explanations
- Provide reassurance and comfort
- Use short sentences
- Repeat information if needed
- Never rush or overwhelm"""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation turns
    for utt in utterances:
        if utt["role"] == "user":
            messages.append({"role": "user", "content": utt["text"]})
        else:
            messages.append({"role": "assistant", "content": utt["text"]})
    
    return {"messages": messages, "metadata": {"has_dementia": is_dementia}}


def process_pitt_corpus(pitt_dir: Path, output_file: Path):
    """Process all .cha files in Pitt corpus"""
    
    samples = []
    
    # Process Dementia group
    dementia_dir = pitt_dir / "Dementia" / "cookie"
    if dementia_dir.exists():
        print(f"Processing Dementia files from {dementia_dir}...")
        for cha_file in dementia_dir.glob("*.cha"):
            utterances = extract_utterances(cha_file)
            if utterances:
                sample = create_training_sample(utterances, is_dementia=True)
                if sample:
                    samples.append(sample)
        print(f"  Processed {len([f for f in dementia_dir.glob('*.cha')])} Dementia files")
    
    # Process Control group
    control_dir = pitt_dir / "Control" / "cookie"
    if control_dir.exists():
        print(f"Processing Control files from {control_dir}...")
        for cha_file in control_dir.glob("*.cha"):
            utterances = extract_utterances(cha_file)
            if utterances:
                sample = create_training_sample(utterances, is_dementia=False)
                if sample:
                    samples.append(sample)
        print(f"  Processed {len([f for f in control_dir.glob('*.cha')])} Control files")
    
    # Write to JSONL
    print(f"\nWriting {len(samples)} samples to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"✓ Done! Created {output_file}")
    print(f"  Total samples: {len(samples)}")
    
    # Show sample
    if samples:
        print("\nSample entry:")
        print(json.dumps(samples[0], indent=2))


if __name__ == "__main__":
    import sys
    
    # Get paths
    if len(sys.argv) > 1:
        pitt_dir = Path(sys.argv[1])
    else:
        pitt_dir = Path("data/Pitt")
    
    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])
    else:
        output_file = Path("data/pitt_corpus_training.jsonl")
    
    if not pitt_dir.exists():
        print(f"Error: Pitt directory not found: {pitt_dir}")
        sys.exit(1)
    
    process_pitt_corpus(pitt_dir, output_file)
