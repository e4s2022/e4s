conda config --set channel_priority flexible
conda env create -f e4s_env.yaml
pip install gdown
gdown 1cyJTYRO5G4kcugAcgSJ7cMsE96GzV_hq -O ./pretrained_ckpts/e4s/iteration_300000.pt
gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O ./pretrained_ckpts/face_parsing/79999_iter.pth
gdown 1FJDN1edNpUyx8Bv5Eo7JutAvbL9zFfn4 -O ./pretrained_ckpts/face_parsing/segnext.small.best_mIoU_iter_140000.pth
gdown 1YL4VuCBhhl-sjI3oPZOJhxTf9rIJRxD5 -O ./pretrained_ckpts/face_parsing/segnext.base.best_mIoU_iter_140000.pth
curl -L -o ./pretrained_ckpts/facevid2vid/00000189-checkpoint.pth.tar "https://download940.mediafire.com/xkjogf9xeojgeXtYFplm5bCoFUHGoCkePfSQ5w3wLkzldJBcTeu3Pm5STboTYF0ZVFdrkS7yKF7pdtWvqamfunG_m4LylmUFin1jXiG657tfjCw05Qrq_vq2fQUprDD2Cr-Id-CtQ8KqhelOZlJT_pmJOMPijovDDI_omhYqrhpU7_0/f0utmig51dal20b/00000189-checkpoint.pth.tar"
curl -L -o ./pretrained_ckpts/facevid2vid/vox-256-spade.yaml "https://download1327.mediafire.com/a7ro467mjvzgyGvE7lLtqoZG25K_1E9fKHd33JAeueq5FdNpPY_KkU62ibabqhUaPCS_Ax7Fh4vyA3vGayck2ZXO0mF11rReowM5GvOLyiAIeWhXW3qg_LIqoL1UAxHEaBsk7IzP5PgNcYrlOyYOPt0mfXRN6R6jB_t7lYjyJ7_pdGk/lpxqkxkktmnj129/vox-256-spade.yaml"
cd pretrained_ckpts/gpen
sh fetch_gepn_models.sh
