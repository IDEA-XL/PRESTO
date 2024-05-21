#!/bin/bash

# Array of prefix words to remove
prefix_words=(
  " "
  "Sure. A potential answer could be: "
  "A possible catalyst can be "
  "A probable catalyst could be "
  "The product can be "
  "A probable product could be "
  "The catalyst can be "
  "A probable product: "
  "Sure. A potential product: "
  "A possible product can be "
  "A possible product could be "
  "The potential reactants: "
  "OK. The reactants may be "
  "Possible reactant(s): "
  "Here are possible reactants: "
  "A possible reagents can be "
  "The reagents can be "
  "A probable reagents could be "
  "A possible solvent can be "
  "A probable solvent could be "
  "The solvent can be "
)

for file in epoch-*/*.txt; do
  # Check if the file exists
  if [ -e "$file" ]; then
    # Create a temporary file to store the modified contents
    temp_file="$(mktemp)"
    
    # Read the file line by line
    while IFS= read -r line; do
      # Iterate over each prefix word
      for prefix in "${prefix_words[@]}"; do
        # Check if the line starts with the prefix
        if [[ "$line" == "$prefix"* ]]; then
          # Remove the prefix from the line
          line="${line/$prefix/}"
        fi
      done
    
        # Check if the line ends with ' .'
        if [[ "$line" == *" ." ]]; then
            # Remove ' .'
            line="${line% .}"
            echo $line
        fi
      
      # Append the modified line to the temporary file
      echo "$line" >> "$temp_file"
    done < "$file"
    
    # Overwrite the original file with the modified contents
    mv "$temp_file" "$file"
    
    echo "Processed and modified $file"
  fi
done