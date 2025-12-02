"""
Convert ALL Pitt Corpus CHAT files to JSONL format for fine-tuning
Processes ALL .cha files from all tasks (cookie, fluency, recall, sentence)
"""

import json
import re
from pathlib import Path
from typing import List, Dict
import sys

def clean_text(text: str) -> str:
    """Clean text by removing timestamps, markers, and special characters"""
    # Remove timestamps (e.g., 1360_2530)
    text = re.sub(r'\d+_\d+', '', text)
    
    # Remove brackets with content [+ exc], [//], etc.
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove special markers (+/, +//, +..., etc.)
    text = re.sub(r'\+[\/\.\,\w]*', '', text)
    
    # Remove hesitation markers (&-um, &-uh, &+flow)
    text = re.sub(r'&[-\+]\w+', '', text)
    text = re.sub(r'&\w+', '', text)
    
    # Remove angle brackets <text>
    text = re.sub(r'<.*?>', '', text)
    
    # Remove @o, @k and other @ markers
    text = re.sub(r'@\w+', '', text)
    
    # Remove (.) pauses
    text = re.sub(r'\(\.\)', '', text)
    text = re.sub(r'\(\.\.+\)', '', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove trailing punctuation at start/end
    text = text.strip('. \t\n\r')
    
    return text


def extract_utterances(cha_file: Path) -> List[Dict[str, str]]:
    """Extract participant utterances from a .cha file"""
    utterances = []
    
    try:
        with open(cha_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"  Error reading {cha_file.name}: {e}")
        return utterances
    
    # Extract lines with *PAR: (participant) and *INV: (investigator)
    lines = content.split('\n')
    
    for line in lines:
        if line.startswith('*INV:'):
            # Extract investigator question/prompt
            text = line.split('*INV:')[1].strip()
            text = clean_text(text)
            
            if text and len(text) > 2:
                utterances.append({"role": "user", "content": text})
        
        elif line.startswith('*PAR:'):
            # Extract participant response
            text = line.split('*PAR:')[1].strip()
            text = clean_text(text)
            
            if text and len(text) > 2:
                utterances.append({"role": "assistant", "content": text})
    
    return utterances


def create_training_sample(utterances: List[Dict], is_dementia: bool, task_type: str) -> Dict:
    """Create a training sample from utterances"""
    
    if not utterances or len(utterances) < 2:
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
        messages.append({
            "role": utt["role"],
            "content": utt["content"]
        })
    
    return {
        "messages": messages,
        "metadata": {
            "has_dementia": is_dementia,
            "task": task_type
        }
    }


def process_task_folder(task_dir: Path, is_dementia: bool, task_name: str) -> List[Dict]:
    """Process all .cha files in a task folder"""
    samples = []
    
    if not task_dir.exists():
        return samples
    
    cha_files = list(task_dir.glob("*.cha"))
    print(f"  Processing {len(cha_files)} files from {task_name}...")
    
    for cha_file in cha_files:
        utterances = extract_utterances(cha_file)
        if utterances:
            sample = create_training_sample(utterances, is_dementia, task_name)
            if sample:
                samples.append(sample)
    
    return samples


def process_all_pitt_data(pitt_dir: Path, output_file: Path):
    """Process ALL .cha files from ALL tasks in Pitt corpus"""
    
    all_samples = []
    tasks = ["cookie", "fluency", "recall", "sentence"]
    
    print("="*60)
    print("Converting ALL Pitt Corpus Data to JSONL")
    print("="*60)
    
    # Process Dementia group
    print("\n[1/2] Processing DEMENTIA group...")
    dementia_samples = 0
    for task in tasks:
        task_dir = pitt_dir / "Dementia" / task
        samples = process_task_folder(task_dir, is_dementia=True, task_name=task)
        all_samples.extend(samples)
        dementia_samples += len(samples)
        print(f"    ✓ {task}: {len(samples)} conversations")
    
    print(f"  Total Dementia samples: {dementia_samples}")
    
    # Process Control group
    print("\n[2/2] Processing CONTROL group...")
    control_samples = 0
    for task in tasks:
        task_dir = pitt_dir / "Control" / task
        samples = process_task_folder(task_dir, is_dementia=False, task_name=task)
        all_samples.extend(samples)
        control_samples += len(samples)
        print(f"    ✓ {task}: {len(samples)} conversations")
    
    print(f"  Total Control samples: {control_samples}")
    
    # Write to JSONL
    print(f"\n{'='*60}")
    print(f"Writing {len(all_samples)} samples to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"{'='*60}")
    print(f"✅ SUCCESS!")
    print(f"{'='*60}")
    print(f"Total samples created: {len(all_samples)}")
    print(f"  - Dementia group: {dementia_samples}")
    print(f"  - Control group: {control_samples}")
    print(f"Output file: {output_file}")
    print(f"File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    print(f"{'='*60}")
    
    # Show sample
    if all_samples:
        print("\n📝 Sample entry (first conversation):")
        print("-"*60)
        sample = all_samples[0]
        print(f"Task: {sample['metadata']['task']}")
        print(f"Has Dementia: {sample['metadata']['has_dementia']}")
        print(f"Messages: {len(sample['messages'])}")
        print("\nFirst few messages:")
        for i, msg in enumerate(sample['messages'][:4]):
            print(f"  [{msg['role']}]: {msg['content'][:80]}...")
            if i >= 3:
                break
        print("-"*60)


if __name__ == "__main__":
    # Get paths
    if len(sys.argv) > 1:
        pitt_dir = Path(sys.argv[1])
    else:
        pitt_dir = Path("data/Pitt")
    
    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])
    else:
        output_file = Path("data/pitt_full_dataset.jsonl")
    
    # Validate
    if not pitt_dir.exists():
        print(f"❌ Error: Pitt directory not found: {pitt_dir}")
        sys.exit(1)
    
    # Process
    process_all_pitt_data(pitt_dir, output_file)
    
    print("\n🎉 Ready for training!")
    print("\nRun training with:")
    print(f"python src/fine_tune.py \\")
    print(f"    --model_path meta-llama/Meta-Llama-3-8B-Instruct \\")
    print(f"    --data_path {output_file} \\")
    print(f"    --use_qlora \\")
    print(f"    --batch_size 1 \\")
    print(f"    --num_epochs 3")
