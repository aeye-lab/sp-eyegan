 # this script assumes that you have setup a ssh key in github.
 # for more information how to setup
 # see: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account

#!/usr/bin/env bash
set -euxo pipefail
 echo 'Installing python requirements...'
 if !(pip install -r requirements.txt);
 then
   echo 'Failed during requirements installation, precisely see error above'
   exit 1
 fi
 for script in get_data create_event_data train_sp_eyegan create_synthetic_data pretrain_constrastive_learning evaluate_downstream
 do
   if !(source scripts/$script.sh);
   then
     echo "Failed during $script, precisely see error above"
     exit 1
   fi
 done
