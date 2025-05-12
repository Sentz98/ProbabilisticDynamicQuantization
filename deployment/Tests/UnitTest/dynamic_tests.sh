#!/bin/sh

# List of folders to delete
FOLDERS_TO_DELETE="
TestCases/TestData/dynamic
PregeneratedData/dynamic
"

# Step 1: Delete the folders in the list
echo "$FOLDERS_TO_DELETE" | while read -r FOLDER; do
    if [ -n "$FOLDER" ]; then
        rm -rf "$FOLDER"

        # Check if the folder was successfully deleted
        if [ ! -d "$FOLDER" ]; then
            echo "Folder deleted successfully: $FOLDER"
        else
            echo "Failed to delete the folder: $FOLDER"
            exit 1
        fi
    fi
done

# Step 2: Generate Data
echo "GENERATE DATA"
#DYNAMIC CONV CONF: in_ch, out_ch, x_in, y_in, w_x, w_y, stride_x, stride_y
python generate_test_data.py --dataset dynamic -t conv_dyn --dynconv 3 8 3 3 3 3 1 1

# Check if the Python script ran successfully
if [ $? -eq 0 ]; then
    echo "Python script executed successfully."
else
    echo "Python script failed to execute."
    exit 1
fi

# Step 3: Build and run CONV test
bash build_and_run_tests.sh -c cortex-m3 -o '-O3'
# Check if the shell script ran successfully
if [ $? -eq 0 ]; then
    echo "Shell script executed successfully."
else
    echo "Shell script failed to execute."
    exit 1
fi

echo "All tasks completed successfully."



# # Step 2: Run another shell script
# echo "Running another shell script: $ANOTHER_SH_SCRIPT"
# bash "$ANOTHER_SH_SCRIPT"



# # Step 3: Run a Python script
# echo "Running Python script: $PYTHON_SCRIPT"
# python3 "$PYTHON_SCRIPT"

# # Check if the Python script ran successfully
# if [ $? -eq 0 ]; then
#     echo "Python script executed successfully."
# else
#     echo "Python script failed to execute."
#     exit 1
# fi

# echo "All tasks completed successfully."