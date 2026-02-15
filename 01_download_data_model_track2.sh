#!/bin/bash

set -e

source env.sh


# Define language codes and their download URLs
declare -A lang_urls=(
    ["en"]="https://duke.app.box.com/shared/static/19ckgfo06hwkorermjejsb05in65js3n"
    ["de"]="https://duke.app.box.com/shared/static/zt36nglx7axehrty2uzi3vn007q9rqkk"
    ["ja"]="https://duke.app.box.com/shared/static/rnj5jz5qrg7wrupo2c3wnk8gqu1ndwxv"
    ["cn"]="https://duke.app.box.com/shared/static/h912rd8jzzh13ywqvsi9yk78zwf1zi3x"
    ["fr"]="https://duke.app.box.com/shared/static/l125t9a9pr2c26r3or6eord3h2z17e8u"
    ["es"]="https://duke.app.box.com/shared/static/vq1o1r42xjwvhnvllfz1vioyis6x6394"
)

# Check if all language directories exist
all_exist=true
for lang in de ja cn fr es en; do
    if [ ! -d "corpora/$lang" ]; then
        all_exist=false
        break
    fi
done


# Download MLS for each language
if [ "$all_exist" = false ]; then
    echo "Download MLS for languages: de, ja, cn, fr, es, en..."
    mkdir -p corpora
    cd corpora
    
    for lang in de ja cn fr es en; do
        lang_dir="$lang"
        lang_file="${lang}.zip"
        lang_url="${lang_urls[$lang]}"
        
        # Skip if directory already exists
        if [ -d "$lang_dir" ]; then
            echo "Directory corpora/$lang_dir already exists, skipping $lang"
            continue
        fi
        
        # Download if file doesn't exist
        if [ ! -f "$lang_file" ]; then
            echo "Downloading $lang..."
            wget -O "$lang_file" "$lang_url"
        else
            echo "File $lang_file already exists, skipping download for $lang"
        fi
        
        # Extract the archive
        if [ -f "$lang_file" ]; then
            echo "Unpacking $lang..."
            # unzip
            if unzip -q "$lang_file" 2>/dev/null; then
                echo "Successfully extracted $lang using unzip"
                
                # Organize extracted files into language directory
                # Check if the language directory already exists (created by extraction)
                if [ ! -d "$lang_dir" ]; then
                    # Find the extracted directory or files
                    extracted_dirs=$(find . -maxdepth 1 -type d ! -name . ! -name "$lang_dir" ! -name "*.zip" | head -1)
                    if [ ! -z "$extracted_dirs" ]; then
                        # Rename the extracted directory to language code
                        mv "$extracted_dirs" "$lang_dir"
                        echo "Renamed extracted directory to $lang_dir"
                    else
                        # If no directory was created, create one and move files
                        mkdir -p "$lang_dir"
                        find . -maxdepth 1 -type f ! -name "$lang_file" ! -name "*.zip" -exec mv {} "$lang_dir/" \; 2>/dev/null || true
                        find . -maxdepth 1 -type d ! -name . ! -name "$lang_dir" ! -name "*.zip" -exec mv {} "$lang_dir/" \; 2>/dev/null || true
                        echo "Created $lang_dir and moved extracted files"
                    fi
                fi
            else
                echo "Warning: Could not extract $lang_file. Please check the file format."
                continue
            fi
        fi
    done
    
    cd ../
fi

model=asv_ssl
if [ ! -d "exp/$model" ]; then
    mkdir -p exp
    cd exp
    if [ ! -f .${model}.zip ]; then
        echo "Download pretrained $model models pre-trained..."
        wget -O ${model}.zip https://duke.app.box.com/shared/static/na6grb7akap4ze66stiazp2azw4zb1f1
        mv ${model}.zip .${model}.zip
    fi
    echo "Unpacking pretrained evaluation models"
    unzip .${model}.zip
    cd ../
fi

check_data=data/cn_dev_enrolls
if [ ! -d $check_data ]; then
    if  [ ! -f .mls_langs.zip ]; then
        echo "Download MLS kaldi format datadir..."
        wget -O mls_langs.zip https://duke.app.box.com/shared/static/vby1xgcdeg4vecdhjsinwcglblqlsd4v
        mv mls_langs.zip .mls_langs.zip
    fi
    echo "Unpacking .mls_langs.zip"
    unzip .mls_langs.zip
fi
