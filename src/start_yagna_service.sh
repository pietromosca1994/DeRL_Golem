export YAGNA_APPKEY=ccd959c719e74cb88cc40061367a6061

echo '[INFO] Starting yagna Daemon'
yagna service run & \
echo '[INFO] Getting currencies from faucet for Golem Testnet'
yagna payment fund
yagna payment status
echo '[INFO] Enable yagna Daemon as a requestor'
yagna payment init --sender