# name of FINN build Python script
build_file=$1

# import the functions from verification_funcs.py that was copied into build folder
sed -i '/os.makedirs("release", exist_ok=True)/a\
from verification_funcs import (\
    create_logger,\
    set_verif_steps,\
    set_verif_io,\
    set_verif_input,\
    set_verif_output,\
    verify_build_output,\
)\
create_logger()\
' $build_file

# set the verification steps, input and expected output npy
# bnn-pynq build config has more indents in code, so account for this
if grep -q "tfc-w1a1" $build_file; then
    sed -i '/output_dir="output_%s_%s" % (model_name, release_platform_name),/a\
            verify_steps=verif_steps,\
            verify_input_npy=set_verif_input(model_name),\
            verify_expected_output_npy=set_verif_output(model_name),\
            verify_save_full_context=True,' $build_file

    sed -i '/build.build_dataflow_cfg(model_file, cfg)/a\
        verify_build_output(cfg, model_name)' $build_file
else # for all other models
    sed -i '/output_dir="output_%s_%s" % (model_name, release_platform_name),/a\
        verify_steps=set_verif_steps(),\
        verify_input_npy=set_verif_input(model_name),\
        verify_expected_output_npy=set_verif_output(model_name),\
        verify_save_full_context=True,' $build_file

    sed -i '/build.build_dataflow_cfg(model_file, cfg)/a\
    verify_build_output(cfg, model_name)' $build_file
fi

# append code at end to remove the copied files and the 
# modified build.py script after build is complete
echo '
os.remove("verification_funcs.py")
os.remove("insert_verif.sh")
import sys
build_file = sys.argv[0]
build_file_original = build_file.replace(".py", "_orig.py")
os.system("mv %s %s" % (build_file_original, build_file))' >> "${build_file}"


