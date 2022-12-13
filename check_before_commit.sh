exec < /dev/tty
# pytest test/ --profile
echo "Did all tests pass? ([no]/yes)"
read input
if [ -z $input ] || [ $input != "yes" ]
then
    echo "Do you want to continue nonetheless? ([no]/yes)"
    read input
    if [ -z $input ] || [ $input != "yes" ]
    then
        exit 1
    fi
fi
pylint --rcfile .pylintrc dptraining/ --load-plugins=perflint
echo "Did the code receive a perfect score (10.00/10)? ([no]/yes)"
read input
if [ -z $input ] || [ $input != "yes" ]
then
    echo "Do you want to continue nonetheless? ([no]/yes)"
    read input
    if [ -z $input ] || [ $input != "yes" ]
    then
        exit 1
    fi

fi
echo "Did you write a test case for everything you modified? ([no]/yes)"
read input
if [ -z $input ] || [ $input != "yes" ]
then
    echo "Do you want to continue nonetheless? ([no]/yes)"
    read input
    if [ -z $input ] || [ $input != "yes" ]
    then
        exit 1
    fi
fi
echo "Do you want to commit your changes? ([no]/yes)"
read input
if [ -z $input ] || [ $input != "yes" ]
then
    exit 1
fi
