#! /usr/bin/bash

generate_random_id() {
    id=$(head /dev/urandom | tr -dc 'A-Za-z0-9' | head -c 4)
    id="${id}_$(head /dev/urandom | tr -dc 'A-Za-z0-9' | head -c 3)"
    echo "${id}"
}

echo $(generate_random_id)


