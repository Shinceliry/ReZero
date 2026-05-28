#!/usr/bin/env bash

cd environments/mic
docker compose build invisiblemic
docker compose up invisiblemic -d
docker compose exec invisiblemic bash