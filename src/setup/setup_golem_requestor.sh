# check yagna version
echo '[INFO] check yagna version'
yagna --version
echo 'Yagna version should be 0.9.2'
# check gfpt version
echo '[INFO] check gfpt version'
gftp --version
echo 'gfpt version should be 0.9.2'
# request api key
yagna app-key create requestor
# list app-key 
yagna app-key list --json > yagna_app_key.json


