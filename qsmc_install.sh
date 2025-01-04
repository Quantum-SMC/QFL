# download QMP-SPDZ
git clone https://github.com/diogoftm/QMP-SPDZ.git
cd QMP-SPDZ
git checkout qdev
git checkout 1c6f9b9aefd2c593b362c24aa6b7e9de9f3878ff
git submodule update --init --recursive
cd ..

rsync -av ./qsmc/ ./QMP-SPDZ/

rm -r qsmc
mv QMP-SPDZ qsmc

# setup QSMC
cd qsmc/Scripts/
chmod +x tldr.sh
./tldr.sh

