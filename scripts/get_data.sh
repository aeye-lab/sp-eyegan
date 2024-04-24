if !(git clone https://github.com/aeye-lab/ecml-ADHD data/ecml-ADHD);
then
    echo 'Failed to clone ADHD data, precise error see above'
fi
for dataset in GazeBase JuDo1000 GazeOnFaces SBSAT
do
    echo "Getting $dataset...."
    if  !(python -m sp_eyegan.get_data --dataset-name $dataset)
    then
        echo "Failed to get $dataset data, precise error see above"
    fi
    echo "Finished $dataset!"
done
