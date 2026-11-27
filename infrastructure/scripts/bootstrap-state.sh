#!/usr/bin/env bash
# =============================================================================
# Bootstrap remote state для Terraform в Yandex Object Storage.
# Создаёт отдельный бакет (не управляемый самим Terraform — против цикла),
# SA для backend и выдаёт статические ключи для backend-конфигурации.
#
# Использование:
#   ./bootstrap-state.sh <folder_id> <environment>
# =============================================================================
set -euo pipefail

FOLDER_ID="${1:?usage: bootstrap-state.sh <folder_id> <environment>}"
ENV="${2:-staging}"
BUCKET="credit-scoring-tfstate-${ENV}"

command -v yc >/dev/null || { echo "Нужен yc CLI (yc)"; exit 1; }

echo ">>> Создаю сервисный аккаунт tf-backend-${ENV}"
SA_ID=$(yc iam service-account create --name "tf-backend-${ENV}" \
  --folder-id "${FOLDER_ID}" --format json | jq -r '.id')

echo ">>> Выдаю роли storage.editor + kms для ${SA_ID}"
yc resource-manager folder add-access-binding "${FOLDER_ID}" \
  --role storage.editor --service-account-id "${SA_ID}" >/dev/null
yc resource-manager folder add-access-binding "${FOLDER_ID}" \
  --role kms.keys.encrypterDecrypter --service-account-id "${SA_ID}" >/dev/null

echo ">>> Статический ключ"
KEY_JSON=$(yc iam access-key create --service-account-id "${SA_ID}" --format json)
ACCESS_KEY=$(echo "${KEY_JSON}" | jq -r '.access_key.id')
SECRET_KEY=$(echo "${KEY_JSON}" | jq -r '.secret')

echo ">>> Бакет ${BUCKET}"
yc storage bucket create --name "${BUCKET}" \
  --default-storage-class standard --public-read false >/dev/null || \
  echo "Бакет уже существует — ок"

echo ""
echo "=== Готово. Выполните перед terraform init: ==="
cat <<EOF
export AWS_ACCESS_KEY_ID="${ACCESS_KEY}"
export AWS_SECRET_ACCESS_KEY="${SECRET_KEY}"
cd infrastructure/environments/${ENV}
terraform init -reconfigure
EOF
echo ""
echo "!!! Секрет сохранён в переменных окружения текущей сессии — НЕ коммитить."
