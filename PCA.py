"""
FILE: PCA.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: classes and functions for PCA.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""


import cPickle
import shelve
import os
import subprocess
import sys
import scipy
from scipy import io
from scipy.linalg import eig, svd

try:
	from matplotlib import pyplot as plot
	import matplotlib as mpl
except ImportError:
	print('No Matplotlib, plotting functions will not work.')

def loadPrincipalComponents( filename ):
	pc = PrincipalComponents()
	pc.load(filename)
	return pc

def loadIndependentComponents( filename ):
	try:
		s = shelve.open( filename, 'r' )
	except:
		raise IOError, 'unable to open', filename
	else:
		ic = IndependentComponents( )
		return ic
#======================================================================#

class PrincipalComponents( object ):
	""" class for storing principal components, weight and the mean
	"""
	def __init__( self, mean=None, weights=None, modes=None, SD=None, projectedWeights=None, sizes=None ):
		self.mean = mean
		self.weights = weights
		self.modes = modes	
		self.SD = SD
		self.sdNorm = 0
		self.projectedWeights = projectedWeights
		self.sizes = sizes
	
	def save(self, filename):
		return self.savePickle(filename)

	def load(self, filename):
		try:
			return self.loadPickle(filename)
		except EOFError:
			return self.loadShelve(filename)

	def saveShelve( self, filename ):
		s = shelve.open( filename+'.pc' )
		s['mean'] = self.mean
		s['weights'] = self.weights
		s['modes'] = self.modes
		s['SD'] = self.SD
		s['sizes'] = self.sizes
		s['projectedWeights'] = self.projectedWeights
		s.close()
		return filename+'.pc'

	def savePickle(self, filename):
		s = {}
		s['mean'] = self.mean
		s['weights'] = self.weights
		s['modes'] = self.modes
		s['SD'] = self.SD
		s['sizes'] = self.sizes
		s['projectedWeights'] = self.projectedWeights
		with open(filename+'.pc', 'w') as f:
			cPickle.dump(s, f)

		return filename+'.pc' 
	
	def loadPickle(self, filename):
		try:
			f = open(filename, 'r')
		except IOError:
			raise IOError, 'unable to open '+filename
		else:
			s = cPickle.load(f)
			self.mean = s['mean']
			self.weights = s['weights']
			self.modes = s['modes']
			self.SD = s['SD']
			if self.SD != None:
				self.sdNorm = True
			
			self.projectedWeights = s.get('projectedWeights')
			self.sizes = s.get('sizes')

	def loadShelve( self, filename ):

		try:
			s = shelve.open( filename, 'r' )
		except ImportError:
			import bsddb3
			_db = bsddb3.hashopen(filename)
			s = shelve.Shelf(_db)
		except:
			raise IOError, 'unable to open '+filename
		else:
			self.mean = s['mean']
			self.weights = s['weights']
			self.modes = s['modes']
			self.SD = s['SD']
			if self.SD != None:
				self.sdNorm = True
			
			self.projectedWeights = s.get('projectedWeights')
			self.sizes = s.get('sizes')
	
	def setProjection( self, P ):
		self.projectedWeights = P
		
	def setMean( self, mean ):
		self.mean = mean

	def setWeights( self, w ):
		""" variance
		"""
		self.weights = w
		
	def setModes( self, m ):
		self.modes = m
		
	def setSD( self, sd ):
		self.SD = sd
		
	def getMean( self ):
		return self.mean.copy()
	
	def getNormSpectrum( self, skipfirst=0 ):
		if skipfirst:
			return self.getWeight()[1:] / self.getWeight()[1:].sum()
		else:
			return self.getWeight() / self.getWeight().sum()
		
	def getWeight( self, n=-1 ):
		if n==-1:
			return self.weights.copy()
		else:
			return self.weights[n].copy()
	
	def getWeightsBySD( self, modes, sd ):
		return scipy.array( sd ) * scipy.sqrt( self.weights[modes] )
		#~ return scipy.array([ sd[i]*scipy.sqrt(self.getWeight(modes[i])) for i in xrange(len(modes)) ])
	
	def calcSDFromWeights( self, modes, w ):
		return scipy.array( w ) / scipy.sqrt( self.weights[modes] )
		
	def getMode( self, n=-1 ):
		if n==-1:
			return self.modes.copy()
		else:
			return self.modes[:,n].copy()

	def getSD( self ):
		return self.SD.copy()
	
	def project( self, data, modes=None, variables=None ):
		""" projects mean centered data onto modes and returns weights
		samples are in columns, just like data matrix.

		modes can be a list of principal component numbers to project 
		only on those components.

		variables can be a list of variable numbers to project using
		only those variables.
		"""

		if modes==None:
			if variables==None:
				F = self.modes
			else:
				F = self.modes[variables,:]
		else:
			if variables==None:
				F = self.modes[:,modes]
			else:
				F = self.modes[variables,:][:,modes]

		# print F.T.shape
		# print data.shape
		# print scipy.array(variables).shape
		
		return scipy.dot(F.T, data)

		# if modes==None:
		# 	return scipy.dot( self.modes.T, data )
		# else:
		# 	return scipy.dot( self.modes[:,modes].T, data )
	
	def reconstruct( self, weights, modes ):
		if len(weights) != len(modes):
			raise ValueError( 'ERROR: PCA.reconstruct: length mismatch between weights and modes: {}, {}'.format(weights, modes) )
		
		f = scipy.array( [ self.getMode(p) for p in modes ] ).T
		
		if self.sdNorm:
			new = ( scipy.dot( f, weights ).squeeze() * self.getSD() + self.getMean() )
		else:
			new = ( scipy.dot( f, weights ).squeeze() + self.getMean() )
		
		return new
	
	def mahalanobis( self, x, modes, centered=False ):
		""" calculates the squared mahalanobis distance of data x wrt to 
		mode numbers defined in list modes.
		
		Subtracts mean from x if centered==False
		"""
		# mean centre
		if not centered:
			xc = x - self.mean
		else:
			xc = x
			
		# project to get weights
		w = self.project( xc, modes )
		
		"""
		# calc m distance
		d = scipy.sqrt(( w * w / self.weights[modes] ).sum())

		return d
		"""
		
		# calc m distance
		d2 = ( w * w / self.weights[modes] ).sum()

		return d2
		
class IndependentComponents( PrincipalComponents ):
	
	def setModes( self, m ):
		# normalise first
		mag = scipy.sqrt((m*m).sum(0))
		self.modes = scipy.transpose( m.T / mag[:,scipy.newaxis] )
	
	def reorderModesByVariance( self, data ):
		proj = self.project( data )
		self.variance = scipy.var( proj, axis=1 )
		self.reorderModesByModeWeights( self.variance )
		
	def reorderModesByModeWeights( self, w ):
		wArgS = scipy.argsort( w )[::-1]
		w = scipy.sort( w )[::-1]
		newModes = []
		for i in wArgS:
			newModes.append( self.modes[:,i] )
		newModes = scipy.transpose(newModes)
		
		self.setModes( newModes )
		self.setWeights( w )
		
class PCList( object ):
	
	def __init__(self, l=None ):
		if l != None:
			self.L = l
		else:
			self.L = []
		
	def append( self, p ):
		self.L.append( p )
		
	def save( self, filename ):
		with open( filename+'.pcl', 'w' ) as f:
			cPickle.dump( self.L, f )
		return filename+'.pcl'
		
	def load( self, filename ):
		with open( filename, 'r' ) as f:
			self.L = cPickle.load( f )
			
	def getModesFracVariance( self, cutoff ):
		pModes = []
		for p in self.L:
			# calculate number of modes to use for profile matching
			cumSpec = p.getNormSpectrum().cumsum()
			pModes.append( scipy.arange( scipy.where(cumSpec>cutoff)[0][0] ) )
			
		return pModes
		
#======================================================================#		
class PCA( object ):
	""" class for performing PCA, and generating data
	
	RENAME to ComponentAnalysis since ICA is also now included
	"""
	def __init__( self, componentType='pca' ):
		self.mean = None
		if componentType=='pca':
			self.PC = PrincipalComponents()
		elif componentType=='ica':
			self.PC = IndependentComponents()
			
		self.data = None
		self.projected = None
		self.sdnorm=False
		
	def setData( self, data, sdnorm=False ):
		""" enter data. variables in rows, observation in columns
		"""

		# calculate mean
		self.PC.setMean( scipy.mean( data, axis=1 ) )
		# centre data
		#~ print 'datashape:', data.shape
		#~ print 'meanshape:', self.getMean().shape
		#~ tempData = data - scipy.row_stack( self.PC.getMean() )
		tempData = scipy.array([data[:,i] - self.PC.getMean() for i in xrange(data.shape[1]) ]).T
		
		# normalise by SD
		if sdnorm:
			self.sdnorm = True
			self.PC.sdNorm = True
			self.PC.setSD( scipy.std( tempData, axis=1 ) )
			#~ tempData = tempData / row_stack( self.PC.getSD() )
			tempData = scipy.array([tempData[:,i] / self.PC.getSD() for i in xrange(tempData.shape[1]) ]).T
			
		self.data = tempData.squeeze()
		
		return 1
	
	def eig_decompose( self ):
		""" decompose input data matrix into principal modes using eigen
		decomposition of covariance matrix
		"""
		
		# calculate covariance
		print 'calculating covariance matrix'
		C = scipy.cov(self.data)
		# eigen decomposition of covariance matrix
		print 'doing eigen decomposition'
		w, modes = eig( C )
		# sort from largest to smallest evalue
		#~ self.weights, self.modes = modeSort( w,e )
		
		self.PC.setWeights( w.astype(float) )
		self.PC.setModes( modes.astype(float) )
		self.PC.setProjection( self.PC.project( self.data )	)

		return 1
	
	def svd_decompose( self ):
		""" decompose input data matrix into principal modes using SVD
		on modified data matrix
		"""
		n = self.data.shape[1]
		
		y = self.data.transpose()/ scipy.sqrt(n - 1)
		u,s,pc = svd( y )
		pc = pc.transpose()
		var = scipy.multiply( s, s )
		
		self.PC.setWeights( var )
		self.PC.setModes( pc[:,:self.data.shape[1] ] )
		self.PC.setProjection( self.PC.project( self.data )	)
		
		return 1 
	
	def lansvd_decompose( self, k, tempFolder, lansvdPath=None ):
		""" decompose input data matrix into principal modes using 
		PROPACK's lansvd implemented in matlab
		"""
		if lansvdPath==None:
			#~ lansvdPath = os.path.expanduser('~/development/PROPACK/lansvd_pca')
			lansvdPath = os.path.expanduser('lansvd_pca')
		
		n = self.data.shape[1]
		y = self.data.transpose()/ scipy.sqrt(n - 1)
		io.savemat( os.path.join(tempFolder,'dataMatrix.mat'), {'A':y, 'K':k} )
		
		retCode = subprocess.call( ['matlab',
									'-nodisplay',
									'-nosplash',
									'-r',
									lansvdPath
									] )
		if retCode != 0:
			raise OSError, 'lansvd failed'
		
		svdOut = scipy.io.loadmat(os.path.join(tempFolder,'svdOutput.mat'))
		s = svdOut['S'].diagonal()
		pc = svdOut['V']
		var = scipy.multiply( s, s )
		#~ pdb.set_trace()
		self.PC.setWeights( var )
		self.PC.setModes( pc[:,:self.data.shape[1] ] )
		self.PC.setProjection( self.PC.project( self.data ) )
	
	#~ def doFastICA( self, fasticaArgs,  ):
		#~ decomposition.fastica( array(self.data.T), **fasticaArgs )
	
	# def mdpPCA( self, **kwargs ):
		
	# 	self.ret = mdp.nodes.PCANode(**kwargs)
	# 	modes = self.ret.execute(self.data.T)
	# 	print 'modes shape:', modes.shape
	# 	self.PC.setWeights( scipy.array(self.ret.d) )
	# 	self.PC.setModes( self.ret.get_projmatrix() )
	# 	self.PC.setProjection( self.ret.get_recmatrix() )
	
	# def mdpNIPALS( self, **kwargs ):
	# 	self.ret = mdp.nodes.NIPALSNode(**kwargs)
	# 	#~ modes = self.ret.execute(self.data.T)
	# 	self.ret.train(self.data.T)
	# 	self.ret.stop_training()
	# 	self.PC.setWeights( scipy.array(self.ret.d) )
	# 	self.PC.setModes( self.ret.get_projmatrix() )
	# 	print 'modes shape:', self.PC.modes.shape
	# 	self.PC.setProjection( self.ret.get_recmatrix() )
	
	def modeSort( self, w, e ):
		wArgS = scipy.argsort(w)[::-1]
		w = scipy.sort(w)[::-1]
		eS = []
		for i in wArgS:
			eS.append( e[:,i] )
		eS = scipy.array(eS)
		
		return w, eS
	
	def generateData( self, pc ):
		""" generate a vector of mesh parameters based on principal
		components and their weights. pc = [ [pc_number, weight], ... ]
		"""
		
		#~ param = self.getMean()
		#~ for p in pc:
			#~ param += p[1]* self.getMode( p[0] )
			#~ 
		#~ return param
		
		new = []
		for p in pc:
			new.append( p[1]* self.getMode( p[0] ) )
		
		if self.sdnorm:	
			return scipy.sum( new, axis=0 )*self.getSD() + self.getMean()
		else:
			return scipy.sum( new, axis=0 ) + self.getMean()

	def unpicklePCs( self, fileName ):
		
		f = open( fileName, 'r' )
		mean, w, modes = cPickle.load(f)
		self.PC.setMean( mean )
		self.PC.setWeights( w )
		self.PC.setModes( modes )
		
		return 1
		
	def projectData( self, modes, newData, reconstruct=True ):
		""" project newData dataset on to the modes defined in list 
		modes, and reconstructs newData in using the define modes if 
		reconstruct is True. Returns projected weights and the 
		reconstructed data vector.
		
		newData needs to have mean removed from it first
		"""
		
		# feature vector
		f = scipy.array( [ self.getMode(p) for p in modes ] ).T
		
		# project to find weights
		w = scipy.dot( f.T, newData )
		
		if reconstruct:
			# reconstruct data
			#~ xNew = ( dot( v, f ) + row_stack( self.getMean() ) ).squeeze()
			xNew = self.reconstruct( w, modes )
			return w, xNew
		else:
			return w
	
	def reconstruct( self, weights, modes ):
		if len(weights) != len(modes):
			raise ValueError( 'ERROR: PCA.reconstruct: length mismatch between weights and pcs' )
		
		f = scipy.array( [ self.getMode(p) for p in modes ] ).T
		
		if self.sdnorm:
			new = ( scipy.dot( f, weights ) * self.getSD() + self.getMean() )
		else:
			if len( weights.shape ) == 1:
				new = ( scipy.dot( f, weights ) + self.getMean() )
			elif len( weights.shape ) == 2:
				new = ( scipy.dot( f, weights ) + self.getMean()[:,scipy.newaxis] )
		
		return new
	
	def getWeightsBySD( self, modes, sd ):
		#~ pdb.set_trace()
		return scipy.array([ sd[i]*scipy.sqrt(self.getWeight(modes[i])) for i in xrange(len(modes)) ])
			
	def getMode( self, n=-1 ):
		return self.PC.getMode( n )
		
	def getWeight( self, n=-1 ):
		return self.PC.getWeight( n )
	
	def getNormSpectrum( self ):
		return self.getWeight() / self.getWeight().sum()
		
	def getMean( self ):
		return self.PC.getMean()	
		
	def getSD( self ):
		return self.PC.getSD()
		
	def getData( self, n ):
		return self.data[:,n].copy()
		
	def mahalanobis( self, x, modes, centered=False ):
		return self.PC.mahalanobis( x, modes, centered )

#======================================================================#
# cross validation

class CrossValidator( object ):
	
	
	def __init__( self, data, nModes, pcaMethod='svd', pcaArgs=None ):
		self.data = data
		self.dataInd = scipy.arange(self.data.shape[1], dtype=int)
		self.pcaMethod = pcaMethod
		self.pcaArgs = pcaArgs
		self.reconModes = scipy.arange( nModes, dtype=int)
		self.PCA = None
		
	def leaveOneOut( self ):

		
		for i in self.dataInd:
			# leave one out
			leftInInd = list(self.dataInd)
			leftInInd.remove(i)
			
			# do pca
			self.PCA = PCA()
			self.PCA.setData( self.data[:,leftInInd] )
			
			if self.pcaMethod=='svd':
				self.PCA.svd_decompose()
			elif self.pcaMethod=='lansvd':
				self.PCA.lansvd_decompose( **self.pcaArgs )
			# elif self.pcaMethod=='mdp':
			# 	self.PCA.mdpPCA( **self.pcaArgs )
				
			# reconstruction
			w = self.PCA.PC.project( self.data[:,i]-self.PCA.getMean(), modes=self.reconModes )
			dr = self.PCA.PC.reconstruct( w, self.reconModes )
			
			error = self.data[:,i] - dr
			
			yield error
			
	def KFold( self, k ):
		"""
		k-fold cross validation. when k=number of data, same as leave one out.
		yields a vector of errors, i.e. is a generator for reconstruction
		errors
		"""
		
		#~ if k==self.data.shape[1]:
			#~ return self.leaveOneOut()
			
		# divide data into k groups
		nPerGroup = int( self.data.shape[1] / k )
		dataInd = scipy.arange(k).repeat( nPerGroup )
		
		for ki in xrange(k):
			
			leftOutInd = scipy.where( dataInd==ki )[0]
			leftInInd = scipy.where( dataInd!=ki )[0]
			
			# do pca
			self.PCA = PCA()
			self.PCA.setData( self.data[:,leftInInd] )
			
			if self.pcaMethod=='svd':
				self.PCA.svd_decompose()
			elif self.pcaMethod=='lansvd':
				self.PCA.lansvd_decompose( **self.pcaArgs )
			# elif self.pcaMethod=='mdp':
			# 	self.PCA.mdpPCA( **self.pcaArgs )
				
			# reconstruction
			errors = []
			for oi in leftOutInd:
				w = self.PCA.PC.project( self.data[:,oi]-self.PCA.getMean(), modes=self.reconModes )
				dr = self.PCA.PC.reconstruct( w, self.reconModes )
				
				errors.append( self.data[:,oi] - dr )
			
			yield (errors, leftInInd)
			
#======================================================================#
# plotting
def plotSpectrum( P, nModes, title, skipfirst=0, cumul=0, PRand=None, barargs={} ):
	
	width = 0.8
	gap = (1 - width)/2
	y = P.getNormSpectrum(skipfirst)[:nModes]*100.0
	if cumul:
		y = y.cumsum()
	left = scipy.arange(len(y))*(width+2*gap) + gap*2 + width/2
	f = plot.figure()
	ax = f.add_subplot(111)
	barplot = ax.bar(left,y, width=width, **barargs )
	# barplot = ax.plot(scipy.arange(len(y))+1,y, linewidth=2 )
	if PRand!=None:
		yRand = PRand.getNormSpectrum(skipfirst)[:nModes]*100.0
		if cumul:
			yRand = yRand.cumsum()
		barplotRand = ax.plot(scipy.arange(len(yRand))+1,yRand, '--', linewidth=2 )
		# barplotRand = ax.bar(left,yRand, width=width)
	else:
		barplotRand=None
	
	ax.set_title( title )
	ax.set_xlabel( 'principal components' )
	ax.set_ylabel( 'percentage of total variation' )
	ax.set_xlim(0.4,len(y)+1)
	#~ ax.set_ylim(0,1.0)
	ax.set_ylim(0,max(y)*1.1)
	ax.set_xticks(range(1, nModes+1))
	ax.set_xticklabels([str(i) for i in range(1, nModes+1)])
	if PRand:
		ax.legend( [barplot[0],barplotRand[0]], ['real data', 'random data'], loc='best' )
	f.show()
	return f,ax,barplot, barplotRand

#======================================================================#
# parallel analysis
def parallelAnalysisPlot( pc ):
	randDataMatrix = scipy.random.random((pc.modes.shape[0],pc.projectedWeights.shape[1]))
	randPCA = PCA()
	randPCA.setData(randDataMatrix)
	randPCA.svd_decompose()
	
	plotObjects = plotSpectrum( pc, 20, '', skipfirst=0, cumul=0, PRand=randPCA.PC )
	
	return randPCA, plotObjects 

def plotModeScatter( pc , xMode=0, yMode=1, zMode=None, pointLabels=None, nTailLabels=3, classes=None, title=None):
	"""
	scatter plot mode projections for up to 3 different modes.
	PointLabels is a list of strings corresponding to each shape.
	nTailLabels defines number of points that are labelled at the tails of the distributions, 
	can be 'all' to label all points. Point labels are for 2D plots only.
	"""

	xWeights = pc.projectedWeights[xMode]
	yWeights = pc.projectedWeights[yMode]
	colourMap = mpl.cm.gray
	if classes==None:
		c = 'r'
	else:
		c = classes
	if zMode == None:
		fig = plot.figure()
		ax = fig.add_subplot(111)
		plt = ax.scatter(xWeights,yWeights, c=c, marker='o', cmap=colourMap)
		if title is None:
			titleString = 'Mode {} vs Mode {}'.format(xMode, yMode)
		else:
			titleString = '{}: Mode {} vs Mode {}'.format(title, xMode, yMode)
		ax.set_title(titleString)
		ax.set_xlabel('Mode %d'%(xMode))
		ax.set_ylabel('Mode %d'%(yMode))
		
		if pointLabels!=None:
			if nTailLabels=='all':
				for label, x, y in zip(pointLabels, xWeights, yWeights):
					plot.annotate(   label, xy=(x,y), xytext=(-5, 5),
							        textcoords='offset points', ha='right', va='bottom',
							        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
							        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
			elif isinstance(nTailLabels, int):
				# sort weights
				xSortedArgs = scipy.argsort(xWeights)
				ySortedArgs = scipy.argsort(yWeights)

				# label x tails
				for i in xSortedArgs[:nTailLabels]:
					plot.annotate(   pointLabels[i], xy=(xWeights[i],yWeights[i]), xytext=(-5, 5),
							        textcoords='offset points', ha='right', va='bottom',
							        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
							        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
				for i in xSortedArgs[-nTailLabels:]:
					plot.annotate(   pointLabels[i], xy=(xWeights[i],yWeights[i]), xytext=(-5, 5),
							        textcoords='offset points', ha='right', va='bottom',
							        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
							        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

				# label y tails
				for i in ySortedArgs[:nTailLabels]:
					plot.annotate(   pointLabels[i], xy=(xWeights[i],yWeights[i]), xytext=(-5, 5),
							        textcoords='offset points', ha='right', va='bottom',
							        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
							        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
				for i in ySortedArgs[-nTailLabels:]:
					plot.annotate(   pointLabels[i], xy=(xWeights[i],yWeights[i]), xytext=(-5, 5),
							        textcoords='offset points', ha='right', va='bottom',
							        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
							        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
			else:
				raise ValueError, "nTailLabels must be 'all' or an integer"

		plot.show()
	else:
		from mpl_toolkits.mplot3d import Axes3D
		fig = plot.figure()
		zWeights = pc.projectedWeights[zMode]
		ax = fig.add_subplot(111, projection='3d')
		plt = ax.scatter(xWeights,yWeights, zWeights, c =c, marker='o', cmap=colourMap)
		if title is None:
			titleString = 'Mode {} vs Mode {} vs Mode {}'.format(xMode, yMode, zMode)
		else:
			titleString = '{}: Mode {} vs Mode {} vs Mode {}'.format(title, xMode, yMode, zMode)
		ax.set_title(titleString)
		ax.set_xlabel('Mode %d'%(xMode))
		ax.set_ylabel('Mode %d'%(yMode))
		ax.set_zlabel('Mode %d'%(zMode))
		plot.show()
	
	return fig, plt
		
	

