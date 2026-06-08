#!/bin/bash
# VoicePrivacy Track 2 submission uploader — Aliyun OSS
#
# Connectivity test:
#   OSS_ACCESS_KEY_ID=XXX OSS_ACCESS_KEY_SECRET=YYY OSS_TEAM=team-PSST1 \
#       ./03_upload_submission_oss_track2.sh test
#
# Upload:
#   OSS_ACCESS_KEY_ID=... OSS_ACCESS_KEY_SECRET=... OSS_TEAM=team-PSST1 \
#       ./03_upload_submission_oss_track2.sh _myanon
#
# Reinstall upload tools: rm .done-upload-tool

set -e

nj=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
[ -f ./env.sh ] && source ./env.sh

##################################################
# OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET — from team_accesskeys.csv
# OSS_TEAM — folder column, e.g. team-PSST1
# OSS_BUCKET / OSS_REGION / OSS_ENDPOINT — override defaults (see team_accesskeys.csv)
# OSS_PARALLEL / OSS_PART_SIZE — multipart upload tuning (large archives)
##################################################
OSS_BUCKET="${OSS_BUCKET:-vpc2026-global}"
OSS_REGION="${OSS_REGION:-ap-southeast-1}"
OSS_ENDPOINT="${OSS_ENDPOINT:-oss-ap-southeast-1.aliyuncs.com}"
OSS_PARALLEL="${OSS_PARALLEL:-32}"
OSS_PART_SIZE="${OSS_PART_SIZE:-64M}"

print_oss_config() {
  echo "    bucket=${OSS_BUCKET} region=${OSS_REGION} endpoint=${OSS_ENDPOINT:-<default>}"
}

run_connectivity_test() {
  print_oss_config

  export OSS_ACCESS_KEY_ID OSS_ACCESS_KEY_SECRET
  local oss_args=(--region "$OSS_REGION")
  [[ -n "$OSS_ENDPOINT" ]] && oss_args+=(-e "$OSS_ENDPOINT")

  local dest_dir="oss://${OSS_BUCKET}/${OSS_TEAM}"
  echo " -- Connectivity test: ${dest_dir}/ --"

  local tmpf
  tmpf=$(mktemp)
  echo "connectivity test $(date)" > "$tmpf"
  local testkey="${dest_dir}/.connectivity_test_$(date +'%Y%m%d_%H%M%S').txt"
  "$OSSUTIL" "${oss_args[@]}" cp -f "$tmpf" "$testkey"
  "$OSSUTIL" "${oss_args[@]}" rm -f "$testkey"
  rm -f "$tmpf"
  echo " -- OK: ${dest_dir}/ is writable --"
}

# Select the anonymization data suffix (or test mode)
if [ -n "$1" ]; then
  anon_suffix=$1
else
  echo "Provide anon_suffix, or 'test' for connectivity check."
  exit 1
fi

if [[ -z "$OSS_ACCESS_KEY_ID" || -z "$OSS_ACCESS_KEY_SECRET" || -z "$OSS_TEAM" ]]; then
  echo "Error: OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET / OSS_TEAM must be set"
  exit 1
fi
OSS_TEAM="${OSS_TEAM// /-}"

# Install ossutil (one time)
mark=.done-upload-tool
if [ ! -f "$mark" ]; then
  echo " == Installing tools to upload dataset =="
  mkdir -p ./utils
  if ! command -v ossutil >/dev/null 2>&1; then
    V=2.2.1
    case "$(uname -s)" in Linux) OS=linux;; Darwin) OS=mac;; *) echo "Unsupported OS"; exit 1;; esac
    case "$(uname -m)" in x86_64|amd64) ARCH=amd64;; aarch64|arm64) ARCH=arm64;; *) echo "Unsupported arch"; exit 1;; esac
    PKG="ossutil-${V}-${OS}-${ARCH}"
    curl -fsSL "https://gosspublic.alicdn.com/ossutil/v2/${V}/${PKG}.zip" -o ./utils/ossutil.zip
    unzip -o -q ./utils/ossutil.zip -d ./utils
    cp "./utils/${PKG}/ossutil" ./utils/ossutil-bin
    chmod +x ./utils/ossutil-bin
  fi
  if ! command -v pigz >/dev/null 2>&1 && command -v micromamba >/dev/null 2>&1; then
    micromamba install -y -c conda-forge pigz pv tar || true
  fi
  touch "$mark"
fi

if command -v ossutil >/dev/null 2>&1; then OSSUTIL=ossutil; else OSSUTIL=./utils/ossutil-bin; fi

# ===== Connectivity test =====
if test "$anon_suffix" = "test"; then
  echo " -- Running connectivity test for team '${OSS_TEAM}' --"
  run_connectivity_test
  exit 0
fi

export OSS_ACCESS_KEY_ID OSS_ACCESS_KEY_SECRET
OSS_ARGS=(--region "$OSS_REGION")
[[ -n "$OSS_ENDPOINT" ]] && OSS_ARGS+=(-e "$OSS_ENDPOINT")
DEST_DIR="oss://${OSS_BUCKET}/${OSS_TEAM}"

# If a yaml config was passed, read the real anon_suffix from it
if [[ "$anon_suffix" == *yaml ]]; then
  echo " -- Config detected, reading 'anon_suffix' --"
  anon_suffix=$(python3 -c "from hyperpyyaml import load_hyperpyyaml; f = open('${anon_suffix}'); print(load_hyperpyyaml(f, None).get('anon_suffix', ''))")
fi
echo " -- Track 2 submission, anon suffix: '${anon_suffix}' --"
print_oss_config

# ===== Collect submission files =====
stuff_to_zip=""
results_exp=exp/results_summary/track2

file=${results_exp}/result_for_rank${anon_suffix}
[ ! -f "$file" ] && echo "File $file does not exist." && exit 1
file=${results_exp}/result_for_submission${anon_suffix}.zip
[ ! -f "$file" ] && echo "File $file does not exist." && exit 1
stuff_to_zip="${stuff_to_zip} ${results_exp}/result_for_rank${anon_suffix} ${results_exp}/result_for_submission${anon_suffix}.zip"

# Track 2 anonymized wav dirs (multilingual dev+test, emodata_track2, multilingual training).
# Size thresholds: rough lower bounds from BM1 baseline (wavs must be present).
tuples=(
  data/en_dev_enrolls${anon_suffix}              266770532
  data/en_dev_trials_mixed${anon_suffix}      1462812299
  data/en_test_enrolls${anon_suffix}            263207110
  data/en_test_trials_mixed${anon_suffix}     1443814112
  data/es_dev_enrolls${anon_suffix}             165950158
  data/es_dev_trials_mixed${anon_suffix}        929122170
  data/es_test_enrolls${anon_suffix}            167971913
  data/es_test_trials_mixed${anon_suffix}       928739516
  data/fr_dev_enrolls${anon_suffix}             169859584
  data/fr_dev_trials_mixed${anon_suffix}        934119914
  data/fr_test_enrolls${anon_suffix}             170716163
  data/fr_test_trials_mixed${anon_suffix}       932888224
  data/de_dev_enrolls${anon_suffix}             240562246
  data/de_dev_trials_mixed${anon_suffix}       1326210127
  data/de_test_enrolls${anon_suffix}            245530566
  data/de_test_trials_mixed${anon_suffix}       1322270368
  data/emodata_track2_dev${anon_suffix}          94933564
  data/emodata_track2_test${anon_suffix}         90957021
  data/train_english${anon_suffix}            8605793658
  data/train_spanish${anon_suffix}            1636411865
  data/train_french${anon_suffix}               2120209608
  data/train_german${anon_suffix}               3123850767
  data/cn_dev_enrolls${anon_suffix}              280080098
  data/cn_dev_trials_mixed${anon_suffix}       1555151489
  data/cn_test_enrolls${anon_suffix}            258619324
  data/cn_test_trials_mixed${anon_suffix}       1441991487
  data/ja_dev_enrolls${anon_suffix}              213357779
  data/ja_dev_trials_mixed${anon_suffix}       1160113571
  data/ja_test_enrolls${anon_suffix}              213360332
  data/ja_test_trials_mixed${anon_suffix}       1167445484
)
length=${#tuples[@]}
for ((i=0; i<length; i+=2)); do
  dir=${tuples[i]}
  [ ! -d "$dir" ] && echo "Directory $dir does not exist." && exit 1
  threshold=${tuples[i+1]}
  dir_size=$(du -sb "$dir" | cut -f1)
  if [ "$dir_size" -lt "$threshold" ]; then
    echo "Directory '$dir' size ($dir_size bytes) is not greater than $threshold bytes. The wavs must be in this folder for submission." && exit 1
  fi
  stuff_to_zip="${stuff_to_zip} ${dir}"
done

# ===== Pack =====
echo " -- Creating submission archive (using: $nj threads, . = progress) --"
archive="submission_track2${anon_suffix}.tar.gz"
if command -v pigz >/dev/null 2>&1; then
  tar --checkpoint=50000 --checkpoint-action=dot \
    --use-compress-program="pigz --best --processes $nj" -cf "$archive" $stuff_to_zip
else
  tar --checkpoint=50000 --checkpoint-action=dot -czf "$archive" $stuff_to_zip
fi
echo

# ===== Upload to OSS =====
remote_name="submission_track2_${OSS_TEAM}${anon_suffix}_$(date +'%Y-%m-%d_%H-%M-%S').tar.gz"
echo " -- Uploading ($(du -sbh "$archive" | cut -f1)) to ${DEST_DIR}/${remote_name}"
echo "    parallel=${OSS_PARALLEL} part-size=${OSS_PART_SIZE} --"
"$OSSUTIL" "${OSS_ARGS[@]}" cp -f \
  --parallel "$OSS_PARALLEL" --part-size "$OSS_PART_SIZE" \
  "$archive" "${DEST_DIR}/${remote_name}"
echo " -- Upload finished: ${DEST_DIR}/${remote_name} --"
