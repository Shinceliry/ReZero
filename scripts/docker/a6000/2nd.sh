#!/usr/bin/env bash

cd environments/A6000_2nd
docker compose build invisiblemic
docker compose up invisiblemic -d
docker compose exec invisiblemic bash