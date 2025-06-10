for lr in 0.0001 0.0003 0.001; do
  for beta in 0.005 0.01 0.02; do
    sed -e "s|learning_rate: .*|learning_rate: $lr|; \
            s|beta: .*|beta: $beta|" \
        ./mlagents_trainer_plugin/a2c/a2c_3DBall.yaml > tmp_config.yaml
    mlagents-learn tmp_config.yaml \
      --run-id lr${lr}_β${beta} \
      --env '/v/Unity/Unity Builds/UnityEnvironment.exe' \
      --train \
      --torch-device cuda:0
  done
done

