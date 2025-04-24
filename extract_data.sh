#!/bin/bash

if [ -z "$1" ]; then
    echo "Uso: $0 <ruta_del_archivo_de_salida>"
    exit 1
fi

output_file="$1"

# Clases disponibles
class_names=("happy" "sad")

# URLs correspondientes (las clases deben estar en el mismo orden que class_names)
class_urls=(
    "https://videos.pexels.com/video-files/6706926/6706926-hd_1920_1080_25fps.mp4 https://videos.pexels.com/video-files/5495175/5495175-uhd_2732_1440_25fps.mp4 https://videos.pexels.com/video-files/5536129/5536129-uhd_1440_2560_25fps.mp4 https://videos.pexels.com/video-files/4584807/4584807-uhd_2560_1440_25fps.mp4 https://videos.pexels.com/video-files/7976476/7976476-uhd_2732_1440_25fps.mp4"
    "https://videos.pexels.com/video-files/5496775/5496775-uhd_2560_1440_30fps.mp4 https://videos.pexels.com/video-files/5981354/5981354-uhd_2732_1440_25fps.mp4 https://videos.pexels.com/video-files/6722759/6722759-uhd_2732_1440_25fps.mp4 https://videos.pexels.com/video-files/8410107/8410107-hd_1920_1080_25fps.mp4 https://videos.pexels.com/video-files/4494789/4494789-hd_1920_1080_30fps.mp4"
)

# Iterar por índice
for i in "${!class_names[@]}"; do
    class_name="${class_names[$i]}"
    urls_string="${class_urls[$i]}"
    echo "Processing class: '$class_name'"
    
    # Convertir la cadena de URLs a array
    read -ra urls <<< "$urls_string"
    for url in "${urls[@]}"; do
        echo "Processing URL: $url for class: $class_name"
        python src/export_training_data.py \
            --class_name "$class_name" \
            --output "$output_file" \
            --source "$url"
    done
done
