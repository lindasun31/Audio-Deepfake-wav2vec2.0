import torchaudio
import os
import matplotlib.pyplot as plt
import math
import torchaudio
import torch
import math


#numpy array is fast for math but for vectorized operations (math on whole arrays at once) 
#in append, numpy recreate the entire array instead of adding the latest tail. for append, better to use list.
#Gather data → use Python lists.
#Do math / transformations → use NumPy arrays.
def plotter(datasets):
    chart_data_name = []
    chart_data_populus = []

    for i, (name, (file_paths, files)) in enumerate(datasets.items(), start=1):
        #just match the shape of the dictionary here (key, (value1, value2))
        chart_data_populus.append(len(files))
        chart_data_name.append(name)
        print(f"jumlah audio {chart_data_name[i-1]}: {chart_data_populus[i-1]}")
        print("Example:")
        for j, f in enumerate(files[:5], start = 1): #loop for displayiing 5 samples
        # j is index, f is files    
            print(f"  {j}. {f}")
            #shape of waveform tensor per samples.
            waveform_path = os.path.join(file_paths, f)
            waveform, sample_rate = torchaudio.load(waveform_path)  
            print(waveform.shape, sample_rate)


        if i % 2 == 0:
            sizes = [chart_data_populus[i-2], chart_data_populus[i-1]]
            labels = [chart_data_name[i-2], chart_data_name[i-1]]

            plt.figure(figsize=(5,5))
            plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
            plt.axis("equal")   # circle shape
            plt.show()
        print("-" * 40)

    # f"{name}" --> f before string: "formatted string"
    # Inside the string, anything in {} is evaluated as Python code.
    #another example with no formatted string:
    #print("jumlah audio", name, ":", len(files))


    #enumerate() is a Python built-in function that adds a 
    #counter to an iterable (like a list, dict, etc.) 
    #so you can loop over both the items and their 
    #index at the same time.

    # f"{name}" --> f before string: "formatted string"
    # Inside the string, anything in {} is evaluated as Python code.
    #another example with no formatted string:
    #print("jumlah audio", name, ":", len(files))


    #enumerate() is a Python built-in function that adds a 
    #counter to an iterable (like a list, dict, etc.) 
    #so you can loop over both the items and their 
    #index at the same time.
    sizes = chart_data_populus
    labels = chart_data_name

    plt.figure(figsize=(5,5))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.axis("equal")   # circle shape
    plt.show()


def audio_slicer(treshold_high, treshold_low, data_set, data_set_name):
    sliced_audio = []

    for i, items in data_set.iterrows():
        waveform, sample_rate = torchaudio.load(data_set.loc[i, 'file path'])
        
        max_samples = int(sample_rate * treshold_high)
        min_samples = int(sample_rate * treshold_low)

        print("audio will be cut to this rate:", max_samples)
        waveform_size = waveform.size(1) #--> get the second column of waveform.size() which is the number of samples in each waveform
        print(f"this audio's duration: {data_set.loc[i, "duration"]}")
        print(f"samples: {waveform_size}")

        chunks = [] #--> nanti isinya audio pendek pendek yg udh di cut per audio panjang

        num_chunks = math.ceil(waveform_size / max_samples)
        #Calculates how many chunks we need to cover the whole audio.
        #ceil rounds up, so the last chunk is included even if smaller than chunk_size.

        for i in range(num_chunks):
            #where the program start and end the cutting
            start = i * max_samples
            end = start + max_samples
            chunk = waveform[:, start:end] #--> slicing the audio. keep in mind to use waveform because waveform is the actual audio (because it has channel and samples)
            chunk_size = chunk.size(1)

            # [:] keep the channel, cut the samples from start to end
            print(f"this chunk's size is {chunk_size}")
            if chunk_size > max_samples: #if above treshold:
                #cut to treshold, make it a new file
                chunk = chunk[:, :max_samples]
                chunks.append(chunk)
                print(f"chunk above max treshold, slicing.")
            elif chunk_size < min_samples: #if below treshold
                #erase/trow away
                print(f"chunk below min treshold, skipping.")
                continue  
                #continue is to skip    
            else:
                #if within treshold, pad to the max samples
                pad_size = max_samples - chunk.size(1)
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))
                chunks.append(chunk)
                print(f"chunk within treshold, padding.")
        sliced_audio.append(chunks)
        print("-"*40)
        
    #checker
    checker = []
    for e, slices in enumerate(sliced_audio):
        sample_sliced = sliced_audio[e]
        checker.append(any(i.size(1) != 96000 for i in sample_sliced)) #check if there exist any non 96000 in a list
    print(f"all {data_set_name} has been sliced to 6 seconds: {all(i == False for i in checker)}")
    return sliced_audio




'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleMLP, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def process_sequence(sequence, max_features=1000):
    """Extract key features from sequence to avoid memory issues"""
    print(f"  Processing sequence type: {type(sequence)}")
    
    if isinstance(sequence, torch.Tensor):
        print(f"  Tensor shape: {sequence.shape}")
        flattened = sequence.flatten().numpy()
        
        # Take only the most important features to save memory
        if len(flattened) > max_features:
            # Option 1: Take first N features
            result = flattened[:max_features]
            print(f"  Reduced from {len(flattened)} to {len(result)} features")
        else:
            result = flattened
            print(f"  Using {len(result)} features")
        return result
        
    elif isinstance(sequence, list):
        print(f"  List length: {len(sequence)}")
        if sequence and isinstance(sequence[0], torch.Tensor):
            # Take mean of each feature dimension instead of flattening everything
            tensors = [t for t in sequence if isinstance(t, torch.Tensor)]
            if tensors:
                # Stack and take mean across time steps
                stacked = torch.stack(tensors)
                result = stacked.mean(dim=0).flatten().numpy()  # Mean pooling
                
                if len(result) > max_features:
                    result = result[:max_features]
                print(f"  Mean pooled to {len(result)} features")
                return result
        
        # Fallback: take first element or small sample
        sample = sequence[:10]  # Take first 10 elements
        result = np.array(sample).flatten()[:max_features]
        print(f"  Sampled to {len(result)} features")
        return result
        
    else:
        result = np.array(sequence).flatten()[:max_features]
        print(f"  Other type sampled to {len(result)} features")
        return result

def prepare_features(df, max_features=1000):
    """Convert dataframe to features with memory optimization"""
    print("Starting feature preparation...")
    features = []
    labels = []
    
    sample_count = min(5, len(df))  # Check first 5 samples
    print(f"Checking first {sample_count} samples to determine feature size...")
    
    # First pass: determine optimal feature size
    feature_sizes = []
    for idx in range(sample_count):
        try:
            sample_feat = process_sequence(df.iloc[idx]['sequence'], max_features)
            feature_sizes.append(len(sample_feat))
            print(f"  Sample {idx}: {len(sample_feat)} features")
        except Exception as e:
            print(f"  Error in sample {idx}: {e}")
    
    if feature_sizes:
        optimal_size = min(max(feature_sizes), max_features)
        print(f"Using feature size: {optimal_size}")
    else:
        optimal_size = max_features
        print(f"Using default feature size: {optimal_size}")
    
    # Second pass: process all data with fixed size
    print("Processing all data...")
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(df)} rows...")
            
        try:
            feature_vector = process_sequence(row['sequence'], optimal_size)
            
            # Ensure consistent size
            if len(feature_vector) < optimal_size:
                # Pad if too small
                feature_vector = np.pad(feature_vector, (0, optimal_size - len(feature_vector)))
            elif len(feature_vector) > optimal_size:
                # Truncate if too large
                feature_vector = feature_vector[:optimal_size]
                
            features.append(feature_vector)
            labels.append(row['label'])
            
        except Exception as e:
            print(f"  ERROR processing row {idx}: {e}")
            continue
    
    print(f"Successfully processed {len(features)} rows")
    print(f"Final feature matrix: {len(features)} x {optimal_size}")
    
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)

def train_mlp(train_df, val_df, hidden_sizes=[256, 128], epochs=500, max_features=1000):
    print("=" * 50)
    print("STARTING MLP TRAINING (USING SEPARATE VALIDATION SET)")
    print("=" * 50)
    
    # Step 1: Prepare training features
    print("Step 1: Preparing training features...")
    X_train, y_train = prepare_features(train_df, max_features)
    
    # Step 2: Prepare validation features
    print("Step 2: Preparing validation features...")
    X_val, y_val = prepare_features(val_df, max_features)
    
    print(f"Training feature matrix shape: {X_train.shape}")
    print(f"Validation feature matrix shape: {X_val.shape}")
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val).reshape(-1, 1)
    
    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    
    # Step 3: Create model
    input_size = int(X_train.shape[1])  # <-- ensure it's a pure Python int
    print(f"Step 3: Creating model with input size: {input_size}")
    model = SimpleMLP(input_size, hidden_sizes, 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    
    print(f"Model: Input({input_size}) -> {hidden_sizes} -> Output(1)")
    
    # Step 4: Training loop
    print("Step 4: Starting training...")
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                predictions = (torch.sigmoid(val_outputs) > 0.5).float()
                val_acc = (predictions == y_val).float().mean()
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc.item():.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"  ↳ New best!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  ↳ Early stopping after {patience} epochs no improvement")
                    break
        
        # Progress indicator
        if (epoch + 1) % 50 == 0:
            print(f"--- Progress: {epoch+1}/{epochs} epochs ---")
    
    print("TRAINING COMPLETED")
    return model, X_train, X_val, y_train, y_val



'''

'''
if __name__ == "__main__":
    print("Initializing MLP training with memory optimization...")
    model, X_train, X_val, y_train, y_val = train_mlp(
        df_train,
        df_val,
        max_features=500
    )
    
    print("Final evaluation...")
    model.eval()
    with torch.no_grad():
        train_pred = torch.sigmoid(model(X_train))
        val_pred = torch.sigmoid(model(X_val))
        
        train_acc = ((train_pred > 0.5).float() == y_train).float().mean()
        val_acc = ((val_pred > 0.5).float() == y_val).float().mean()
        
        print(f"Final Train Accuracy: {train_acc.item():.4f}")
        print(f"Final Val Accuracy: {val_acc.item():.4f}")



'''