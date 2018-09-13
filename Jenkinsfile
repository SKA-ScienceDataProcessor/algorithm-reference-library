#!/usr/bin/env groovy
// Jenkinsfile (Declarative pipeline)
//
// https://jenkins.io/doc/book/pipeline/jenkinsfile/
//
// Stages section:
// - Setup
// - Build
// - Test
// - Deploy
//
// Post section: (https://jenkins.io/doc/book/pipeline/syntax/#post)


pipeline {
    agent {label 'sdp-ci-01'}
    environment {
	MPLBACKEND='agg'
	ARLROOT="${env.WORKSPACE}"
    }
    stages {
// We can skip checkout step if Jenkinsfile is in same repo as the source code (the checkout is configured in Jenkins server, note that we need to enable git-lfs!
//	stage('Checkout'){
//	    steps {
//		echo 'Checking out repository'
//		checkout([$class: 'GitSCM', branches: [[name: '*/master']], doGenerateSubmoduleConfigurations: false, extensions: [[$class: 'CleanBeforeCheckout'], [$class: 'GitLFSPull']], submoduleCfg: [], userRemoteConfigs: [[credentialsId: '2ca2f96d-f272-46d1-accf-8b64a4a0a48e', url: 'https://github.com/mfarrera/algorithm-reference-library']]])
		//checkout scm
//	    }
//        }
        stage('Setup') {
            steps {
		echo "Running ${env.BUILD_ID} on ${env.JENKINS_URL}"
                echo 'Setting up a fresh Python virtual environment...'
		sh '''
		virtualenv -p `which python3` _build
		echo 'Activating virtual environment...'
		source _build/bin/activate
		echo 'Installing requirements'
		pip install -U pip setuptools
		pip install coverage numpy
		pip install virtualenvwrapper
		pip install -r requirements.txt
		echo 'Adding the arl and ffiwrappers path to the virtual environment'
		echo '(equivalent to setting up PYTHONPATH environment variable'
		source virtualenvwrapper.sh
		add2virtualenv $WORKSPACE
		add2virtualenv $WORKSPACE/ffiwrappers/src
		'''
            }
        }
	stage('Build'){
	    steps {
		echo 'Building..'
		sh '''
		source _build/bin/activate
		export LDFLAGS="$(python3-config --ldflags) -lcfitsio"
		python setup.py install
		'''
	    }
	}
        stage('TestARL') {
            steps {
                echo 'Testing ARL..'
		sh '''
		source _build/bin/activate
		export MPLBACKEND=agg
		pip install pytest pytest-xdist pytest-cov
		py.test tests -n 4 --verbose --cov=libs --cov=processing_components --cov=workflows --cov-report=html:coverage tests
		'''
 		//Make coverage report
		//coverage html --include=libs/*,processing_components/*,workflows/* -d coverage
            }
        }
        stage('TestFfiwrappers') {
            steps {
                echo 'Testing ffiwrappers..'
		sh '''
		source _build/bin/activate
		export ARLROOT=$WORKSPACE
		source tests/ffiwrapped/run-tests.sh
		'''
            }
        }
        stage('Deploy') {
            steps {
                echo 'Make documentation....'
		sh '''
		source  _build/bin/activate
		export MPLBACKEND=agg
		make -k -j -C docs html
		'''
		// make -C docs latexpdf # Broken currently?
            }
        }
    }
    post {
        always {
            echo 'FINISHED'
	    slackSend baseUrl: 'https://sdp-execution-engine.slack.com/services/hooks/jenkins-ci/', 
	    channel: '#jenkins', 
	    color: 'good', 
	    message: "Pipeline ${currentBuild.fullDisplayName} ${env.JOB_NAME} ${env.BUILD_NUMBER} completed with Status: ${env.BUILD_STATUS} (<${env.BUILD_URL}|Open>)", 
	    tokenCredentialId: 'a06474f9-0c86-4dc7-a477-42d7d1a1cc71'
        }
    	failure {
             mail to: 'mf582@mrao.cam.ac.uk, pw410@cam.ac.uk, realtimcornwell@gmail.com',
             subject: "Failed Jenkins Pipeline: ${currentBuild.fullDisplayName} Status:${env.BUILD_STATUS} ",
             body: "Something is wrong with ${env.BUILD_URL} Status: ${env.BUILD_STATUS} "
            
    	}
    	fixed {
             mail to: 'mf582@mrao.cam.ac.uk, pw410@cam.ac.uk, realtimcornwell@gmail.com',
             subject: "Jenkins Pipeline is back to normal: ${currentBuild.fullDisplayName} Status:${env.BUILD_STATUS}  ",
             body: "See ${env.BUILD_URL}"
	}
	success {
		sshPublisher alwaysPublishFromMaster: true, 
		publishers: [sshPublisherDesc(configName: 'vm12', 
				transfers: [sshTransfer(excludes: '', 
					execCommand: '', execTimeout: 120000, 
					flatten: false, 
					makeEmptyDirs: false, 
					noDefaultExcludes: false, 
					patternSeparator: '[, ]+', 
					remoteDirectory: 'algorithm-reference-library', 
					remoteDirectorySDF: false, 
					removePrefix: '', 
					sourceFiles: 'docs/build/**'), 
				sshTransfer(excludes: '', 
					execCommand: '', execTimeout: 120000, 
					flatten: false, 
					makeEmptyDirs: false, 
					noDefaultExcludes: false, 
					patternSeparator: '[, ]+', 
					remoteDirectory: 'algorithm-reference-library', 
					remoteDirectorySDF: false, 
					removePrefix: '', 
					sourceFiles: 'coverage/**')], 
				usePromotionTimestamp: false, 
				useWorkspaceInPromotion: false, 
				verbose: false)]

    	}
    }	
}

