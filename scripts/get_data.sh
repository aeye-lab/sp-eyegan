if !(git clone https://github.com/aeye-lab/ecml-ADHD data/ecml-ADHD);
then
    echo 'Failed to clone ADHD data, precise error see above'
    exit 1
fi
for dataset in GazeBase JuDo100 GazeOnFaces SB-SAT
do
    echo "Getting $dataset...."
    if  !(python -m sp_eyegan.get_data --dataset-name $dataset)
    then
        echo "Failed to get $dataset data, precise error see above"
    fi
    echo "Finished $dataset!"
done
