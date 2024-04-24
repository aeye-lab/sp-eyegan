if !(python -m sp_eyegan.create_event_data_from_gazebase --stimulus text);
then
    echo 'Failed to create event data for gazebase, precise error see above'
    exit 1
fi
