exec < /dev/tty
pytest test/ --profile
echo "Did all tests pass? ([no]/yes)"
read input
if [ -z $input ] || [ $input != "yes" ]
then
    echo "Please make sure that all tests pass!"
    exit 1
fi
pylint --rcfile .pylintrc dptraining/ --load-plugins=perflint
echo "Did the code receive a perfect score (10.00/10)? ([no]/yes)"
read input
if [ -z $input ] || [ $input != "yes" ]
then
    echo "Please incorporate all reasonable suggestions of pylint and disable unreasonable ones!"
    exit 1
fi
echo "Did you write a test case for everything you modified? ([no]/yes)"
read input
if [ -z $input ] || [ $input != "yes" ]
then
    echo "Please write test cases for your code!"
    exit 1
fi
echo "Congrats! Your code appears to be ready to commit :)"
