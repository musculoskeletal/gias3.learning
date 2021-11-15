"""
FILE: PCA_fitting.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: functions and classes for mesh/data fitting using principal
components.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""
import logging

import numpy
import sys
from scipy.optimize import leastsq, fmin

from gias3.common import transform3D
from gias3.registration import alignment_fitting

log = logging.getLogger(__name__)


def rigidObj(x, p, func, func_args=()):
    """ x = [tx,ty,tz,rx,ry,rz]
    """
    # print x
    # apply transform to parameters
    # ~ pT = transform3D.transformRigid3D( p, x )
    pT = transform3D.transformRigid3DAboutCoM(p, x)
    # calc error
    # ~ pdb.set_trace()
    return func(pT.T.ravel(), *func_args)


def rigidScaleObj(x, p, func, func_args=()):
    """ x = [tx,ty,tz,rx,ry,rz,s]
    """
    # apply transform to parameters
    pT = transform3D.transformRigidScale3DAboutCoM(p, x)
    # calc error
    return func(pT.T.ravel(), *func_args)


def rigidMode0Obj(x, func, pca, m_weight, func_args=()):
    # reconstruct shape using 1st mode weight
    p = pca.reconstruct(pca.getWeightsBySD([0, ], [x[6], ]), [0, ])
    # apply rigid transform
    # ~ p = transform3D.transformRigid3D( p.reshape((3,-1)).T, x[:6] )
    p = transform3D.transformRigid3DAboutCoM(p.reshape((3, -1)).T, x[:6])
    # calc error
    return func(p.T.ravel(), *func_args) + mahalanobis([x[6], ]) * m_weight


def rigidModeNObj(x, func, pca, modes, m_weight, func_args):
    # reconstruct shape using modes
    p = pca.reconstruct(pca.getWeightsBySD(modes, x[6:]), modes)
    # reconstruct rigid transform
    # ~ p = transform3D.transformRigid3D( p.reshape((3,-1)).T, x[:6] )
    p = transform3D.transformRigid3DAboutCoM(p.reshape((3, -1)).T, x[:6])
    # calc error
    funcErr = func(p.T.ravel(), *func_args)
    MErr = mahalanobis(x[6:]) * m_weight
    Err = funcErr + MErr

    # ~ sys.stdout.write( '\rfunc rms:'+str(scipy.sqrt(funcErr.mean())) )
    # sys.stdout.write( '\rfunc rms:'+str(scipy.sqrt(Err.mean())) )
    # sys.stdout.flush()

    return funcErr + MErr
    # ~ return func( p.T.ravel(), *func_args ) + mahalanobis( x[6:] )*m_weight


def rigidModeNRotateAboutCoMObj(x, func, pca, modes, m_weight, func_args):
    # print 'rigidModeNRotateAboutCoMObj x:', x

    # reconstruct shape using modes
    p = pca.reconstruct(pca.getWeightsBySD(modes, x[6:]), modes)
    # reconstruct rigid transform
    p = transform3D.transformRigid3DAboutCoM(p.reshape((3, -1)).T, x[:6])
    # calc error
    funcErr = func(p.T.ravel(), *func_args)
    MErr = mahalanobis(x[6:]) * m_weight
    Err = funcErr + MErr

    # ~ sys.stdout.write( '\rfunc rms:'+str(scipy.sqrt(funcErr.mean())) )
    # sys.stdout.write( '\rfunc rms:'+str(scipy.sqrt(Err.mean())) )
    # sys.stdout.flush()

    return funcErr + MErr
    # ~ return func( p.T.ravel(), *func_args ) + mahalanobis( x[6:] )*m_weight


def rigidScaleModeNObj(x, func, pca, modes, m_weight, func_args):
    # reconstruct shape using modes
    p = pca.reconstruct(pca.getWeightsBySD(modes, x[7:]), modes)
    # reconstruct rigid transform
    p = transform3D.transformRigidScale3DAboutCoM(p.reshape((3, -1)).T, x[:7])
    # calc error
    funcErr = func(p.T.ravel(), *func_args)
    MErr = mahalanobis(x[7:]) * m_weight
    Err = funcErr + MErr

    # ~ sys.stdout.write( '\rfunc rms:'+str(scipy.sqrt(funcErr.mean())) )
    # sys.stdout.write( '\rfunc rms:'+str(scipy.sqrt(Err.mean())) )
    # sys.stdout.flush()

    return funcErr + MErr


def modeNObj(x, func, pca, modes, rigid_t, func_args):
    # reconstruct shape using modes
    p = pca.reconstruct(pca.getWeightsBySD(modes, x), modes)
    # reconstruct rigid transform
    # ~ p = transform3D.transformRigid3D( p.reshape((3,-1)).T, rigidT )
    p = transform3D.transformRigid3DAboutCoM(p.reshape((3, -1)).T, rigid_t)
    # calc error
    return func(p.T.ravel(), *func_args) + mahalanobis(x[6:])


def mahalanobis(x):
    return numpy.sqrt(numpy.multiply(x, x).sum())


class PCFit(object):
    xtol = 1e-6
    ftol = 1e-6
    maxiter = None
    maxfev = None
    useFMin = False

    def __init__(self, pc=None):
        self.pc = pc

        self.rigidOpt = None
        self.rigidScaleOut = None
        self.rigidMode0Opt = None
        self.rigidModeNOpt = None
        self.rigidScaleModeNOpt = None

    def setPC(self, pc):
        self.pc = pc

    def rigidFit(self, func, x0=None, func_args=(), p0=None):
        if x0 is None:
            x0 = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if p0 is None:
            p0 = self.pc.getMean().reshape((3, -1)).T

        x0 = numpy.array(x0)

        if self.useFMin:
            self.rigidOpt = fmin(rigidObj, x0,
                                 args=(p0, func, func_args),
                                 xtol=self.xtol,
                                 ftol=self.ftol,
                                 maxiter=self.maxiter,
                                 maxfun=self.maxfev,
                                 )
        else:
            self.rigidOpt = leastsq(rigidObj, x0,
                                    args=(p0, func, func_args),
                                    xtol=self.xtol,
                                    ftol=self.ftol,
                                    epsfcn=1e-5
                                    )[0]

        # ~ pOpt = transform3D.transformRigid3D( p0, self.rigidOpt ).T
        pOpt = transform3D.transformRigid3DAboutCoM(p0, self.rigidOpt).T
        return self.rigidOpt, pOpt.ravel()

    def rigidScaleFit(self, func, x0=None, func_args=(), p0=None):
        if x0 is None:
            x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        if p0 is None:
            p0 = self.pc.getMean().reshape((3, -1)).T

        x0 = numpy.array(x0)
        p0 = self.pc.getMean().reshape((3, -1)).T

        if self.useFMin:
            self.rigidScaleOpt = fmin(rigidScaleObj, x0,
                                      args=(p0, func, func_args),
                                      xtol=self.xtol,
                                      ftol=self.ftol,
                                      maxiter=self.maxiter,
                                      maxfun=self.maxfev,
                                      )
        else:
            self.rigidScaleOpt = leastsq(rigidScaleObj, x0,
                                         args=(p0, func, func_args),
                                         xtol=self.xtol,
                                         ftol=self.ftol
                                         )[0]

        pOpt = transform3D.transformRigidScale3DAboutCoM(p0, self.rigidScaleOpt).T
        return self.rigidScaleOpt, pOpt.ravel()

    def rigidMode0Fit(self, func, x0=None, m_weight=0.0, func_args=()):
        if x0 is None:
            x0 = numpy.hstack((self.rigidOpt, 0.0))
        elif len(x0) != 7:
            raise ValueError('x0 must be 7 long')

        if self.useFMin:
            self.rigidMode0Opt = fmin(rigidMode0Obj, x0,
                                      args=(func, self.pc, m_weight, func_args),
                                      xtol=self.xtol,
                                      ftol=self.ftol,
                                      maxiter=self.maxiter,
                                      maxfun=self.maxfev,
                                      )
        else:
            self.rigidMode0Opt = leastsq(rigidMode0Obj, x0,
                                         args=(func, self.pc, m_weight, func_args),
                                         xtol=self.xtol,
                                         ftol=self.ftol
                                         )[0]

        pOpt = self.pc.reconstruct(self.pc.getWeightsBySD([0, ], [self.rigidMode0Opt[6], ]), [0, ])
        # ~ pOpt = transform3D.transformRigid3D( pOpt.reshape((3,-1)).T, self.rigidMode0Opt[:6] )
        pOpt = transform3D.transformRigid3DAboutCoM(pOpt.reshape((3, -1)).T, self.rigidMode0Opt[:6])
        return self.rigidMode0Opt, pOpt.T.ravel()

    def rigidModeNFit(self, func, modes=None, x0=None, m_weight=0.0, maxfev=None, func_args=()):
        # fit modes beyond mode 0
        if x0 is None:
            x0 = numpy.hstack((self.rigidMode0Opt, numpy.zeros(len(modes))))
        if modes is None:
            modes = [1, 2, 3]

        modes = numpy.hstack((0, modes))
        # print '\nx0:', x0
        # print 'm_weight:', m_weight

        if self.useFMin:
            if maxfev is None:
                maxfev = self.maxfev
            self.rigidModeNOpt = fmin(rigidModeNObj, x0,
                                      args=(func, self.pc, modes, m_weight, func_args),
                                      xtol=self.xtol,
                                      ftol=self.ftol,
                                      maxfun=maxfev,
                                      maxiter=self.maxiter,
                                      )
        else:
            if maxfev is None:
                self.rigidModeNOpt = leastsq(rigidModeNObj, x0,
                                             args=(func, self.pc, modes, m_weight, func_args),
                                             xtol=self.xtol,
                                             ftol=self.ftol,
                                             )[0]
            else:
                self.rigidModeNOpt = leastsq(rigidModeNObj, x0,
                                             args=(func, self.pc, modes, m_weight, func_args),
                                             xtol=self.xtol,
                                             ftol=self.ftol,
                                             maxfev=maxfev
                                             )[0]

        pOpt = self.pc.reconstruct(self.pc.getWeightsBySD(modes, self.rigidModeNOpt[6:]), modes)
        pOpt = transform3D.transformRigid3DAboutCoM(pOpt.reshape((3, -1)).T, self.rigidModeNOpt[:6])
        # print '\nxOpt:', self.rigidModeNOpt

        return self.rigidModeNOpt, pOpt.T.ravel()

    def rigidModeNRotateAboutCoMFit(self, func, modes=None, x0=None, m_weight=0.0, maxfev=None, func_args=()):
        # fit modes beyond mode 0
        if x0 is None:
            x0 = numpy.hstack((self.rigidMode0Opt, numpy.zeros(len(modes))))
        if modes is None:
            modes = [1, 2, 3]

        modes = numpy.hstack((0, modes))
        # print '\nrigidModeNRotateAboutCoMFit x0:', x0
        # print 'm_weight:', m_weight
        if self.useFMin:
            if maxfev is None:
                maxfev = self.maxfev
            self.rigidModeNOpt = fmin(rigidModeNRotateAboutCoMObj, x0,
                                      args=(func, self.pc, modes, m_weight, func_args),
                                      xtol=self.xtol,
                                      ftol=self.ftol,
                                      maxiter=self.maxiter,
                                      maxfun=maxfev,
                                      )
        else:
            if maxfev is None:
                self.rigidModeNOpt = leastsq(rigidModeNRotateAboutCoMObj, x0,
                                             args=(func, self.pc, modes, m_weight, func_args),
                                             xtol=self.xtol,
                                             ftol=self.ftol,
                                             )[0]
            else:
                self.rigidModeNOpt = leastsq(rigidModeNRotateAboutCoMObj, x0,
                                             args=(func, self.pc, modes, m_weight, func_args),
                                             xtol=self.xtol,
                                             ftol=self.ftol,
                                             maxfev=maxfev
                                             )[0]

        pOpt = self.pc.reconstruct(self.pc.getWeightsBySD(modes, self.rigidModeNOpt[6:]), modes)
        pOpt = transform3D.transformRigid3DAboutCoM(pOpt.reshape((3, -1)).T, self.rigidModeNOpt[:6])
        # print '\nxOpt:', self.rigidModeNOpt

        return self.rigidModeNOpt, pOpt.T.ravel()

    def rigidScaleModeNFit(self, func, modes=None, x0=None, m_weight=0.0, maxfev=None, func_args=()):
        # fit modes
        if x0 is None:
            x0 = numpy.hstack((self.rigidScaleOpt, numpy.zeros(len(modes))))
        if modes is None:
            modes = [0, 1, 2]

        if self.useFMin:
            if maxfev is None:
                maxfev = self.maxfev
            self.rigidScaleModeNOpt = fmin(rigidScaleModeNObj, x0,
                                           args=(func, self.pc, modes, m_weight, func_args),
                                           xtol=self.xtol,
                                           ftol=self.ftol,
                                           maxfun=maxfev,
                                           maxiter=self.maxiter,
                                           )
        else:
            if maxfev is None:
                self.rigidScaleModeNOpt = leastsq(rigidScaleModeNObj, x0,
                                                  args=(func, self.pc, modes, m_weight, func_args),
                                                  xtol=self.xtol,
                                                  ftol=self.ftol,
                                                  )[0]
            else:
                self.rigidScaleModeNOpt = leastsq(rigidScaleModeNObj, x0,
                                                  args=(func, self.pc, modes, m_weight, func_args),
                                                  xtol=self.xtol,
                                                  ftol=self.ftol,
                                                  maxfev=maxfev
                                                  )[0]

        pOpt = self.pc.reconstruct(self.pc.getWeightsBySD(modes, self.rigidScaleModeNOpt[7:]), modes)
        pOpt = transform3D.transformRigidScale3DAboutCoM(pOpt.reshape((3, -1)).T, self.rigidScaleModeNOpt[:7])
        return self.rigidScaleModeNOpt, pOpt.T.ravel()

    def modeNFit(self, func, modes=None, x0=None, func_args=(), maxfev=None, m_weight=1.0):
        """ fit mode weights only
        """
        if x0 is None:
            x0 = numpy.hstack((self.rigidMode0Opt[-1], numpy.zeros(len(modes))))
        if modes is None:
            modes = [1, 2, 3]

        modes = numpy.hstack((0, modes))
        rigidT = self.rigidMode0Opt[:6]

        if self.useFMin:
            if maxfev is None:
                maxfev = self.maxfev
            self.modeNOpt = fmin(modeNObj, x0,
                                 args=(func, self.pc, modes, m_weight, func_args),
                                 xtol=self.xtol,
                                 ftol=self.ftol,
                                 maxfun=maxfev,
                                 maxiter=self.maxiter,
                                 )
        else:
            self.modeNOpt = leastsq(modeNObj, x0,
                                    args=(func, self.pc, modes, rigidT, func_args),
                                    xtol=self.xtol,
                                    ftol=self.ftol
                                    )[0]

        pOpt = self.pc.reconstruct(self.pc.getWeightsBySD(modes, self.modeNOpt), modes)
        # ~ pOpt = transform3D.transformRigid3D( pOpt.reshape((3,-1)).T, rigidT[:6] )
        pOpt = transform3D.transformRigid3DAboutCoM(pOpt.reshape((3, -1)).T, rigidT[:6])
        return self.modeNOpt, pOpt.T.ravel()


def project3DPointsToSSM(data, SSM, project_modes, project_variables=None,
                         landmark_is=None, init_rotation=None, do_scale=False,
                         verbose=False, ret_t=False):
    # rigid align data to SSM mean data
    if init_rotation is None:
        init_rotation = numpy.array([0.0, 0.0, 0.0])

    # landmarkIs = numpy.array(landmarkIs, dtype=int)

    # fit has to be done on a subset of recon data is projection is on a subset of variables
    if project_variables is not None:
        # meanData = SSM.getMean()[projectVariables].reshape((3,-1)).T
        meanData = SSM.getMean().reshape((3, -1)).T[landmark_is, :]
    else:
        meanData = SSM.getMean().reshape((3, -1)).T

    log.debug('data shape', data.shape)
    log.debug('meandata shape', meanData.shape)
    if landmark_is is not None:
        log.debug('landmarkIs shape', landmark_is.shape)

    if do_scale:
        align1X0 = numpy.hstack([meanData.mean(0) - data.mean(0), init_rotation, 1.0])
        T1, dataT = alignment_fitting.fitRigidScale(data, meanData, align1X0, verbose=verbose)
    else:
        align1X0 = numpy.hstack([meanData.mean(0) - data.mean(0), init_rotation])
        # T1, dataT = alignment_fitting.fitRigid( data, meanData, align1X0, verbose=verbose )
        T1, dataT = alignment_fitting.fitRigid(data, meanData, align1X0, xtol=1e-6, epsfcn=1e-9, verbose=verbose)

    # project data onto principal components
    pcWeights = SSM.project((dataT.T - meanData.T).ravel(), modes=project_modes, variables=project_variables)
    # reconstruction data using projected weights
    reconData = SSM.reconstruct(pcWeights, project_modes).reshape((3, -1)).T

    # fit has to be done on a subset of recon data is projection is on a subset of variables
    if project_variables is not None:
        # reconDataFit = reconData.ravel()[projectVariables].reshape((3,-1)).T
        reconDataFit = reconData.T.ravel().reshape((3, -1)).T[landmark_is, :]
    else:
        reconDataFit = reconData

    # inverse transform back to image
    if do_scale:
        align2X0 = numpy.hstack([reconDataFit.mean(0) - data.mean(0), -init_rotation, 1.0])
        T2, reconDataFitT = alignment_fitting.fitRigidScale(reconDataFit, data, align2X0, verbose=verbose)
    else:
        align2X0 = numpy.hstack([reconDataFit.mean(0) - data.mean(0), -init_rotation])
        # T2, reconDataT = alignment_fitting.fitRigid( reconData, data, align2X0, verbose=verbose )
        T2, reconDataFitT = alignment_fitting.fitRigid(reconDataFit, data, align2X0, xtol=1e-6, epsfcn=1e-9,
                                                       verbose=verbose)

    # transform full recon data to data space if data is a subset of variables
    if project_variables is not None:
        reconDataT = transform3D.transformRigid3DAboutCoM(reconData, T2)
    else:
        reconDataT = reconDataFitT

    if ret_t:
        return pcWeights, reconDataT, dataT, reconData, T2
    else:
        return pcWeights, reconDataT, dataT, reconData


def fitSSMTo3DPoints(data, SSM, fit_modes, fit_point_indices=None, m_weight=0.0,
                     init_rotation=None, do_scale=False,
                     landmark_targets=None, landmark_evaluator=None, landmark_weights=None,
                     recon2coords=None, verbose=False):
    """
    Fit a shape model to a set of correspondent points by optimising
    translation, rotation, and PCA scores.
    
    inputs
    ------
    data: nx3 array of target point coordinates. N Points must correspond to
        points in the shape model
    SSM: a gias2.learning.PCA.PrincipalComponents object
    fitModes: a list of PC modes to fit, e.g. [0,1,2] to fit the 1st 3 modes.
    fitPointIndices: [optional] restrict fitting to certain points in the 
        shape model
    m_weight: [float, optional] mahalanobis weight
    initRotation: [list, optional] initial x,y,z rotation in radians
    doScale: [bool, False] fit for scaling
    
    Returns
    -------
    xOpt: array of optimised parameters. First 3 elements are translation, 
        then 3 for rotation, and the rest are PC scores in terms of standard
        deviations.
    reconDataOptT: fitted shape model points aligned to the mean shape
    dataT: target data points aligned to the mean shape
    reconDataOpt:  fitted shape model points aligned to the target data
    """

    if recon2coords is None:
        # Function to convert ssm data into point coordinates. Default is for
        # fieldwork models
        def recon2coords(xr):
            return xr.reshape((3, -1)).T

    log.debug('fitting SSM to points')
    # rigid align data to SSM mean data
    if init_rotation is None:
        init_rotation = numpy.array([0.0, 0.0, 0.0])
    else:
        init_rotation = numpy.array(init_rotation)

    # fit has to be done on a subset of recon data is projection is on a subset of variables
    if fit_point_indices is not None:
        fit_point_indices = numpy.array(fit_point_indices, dtype=int)
        # meanData = SSM.getMean().reshape(coord_shape).T[fitPointIndices,:]
        meanData = recon2coords(SSM.getMean())[fit_point_indices, :]
    else:
        # meanData = SSM.getMean().reshape(coord_shape).T
        meanData = recon2coords(SSM.getMean())

    # print 'data shape:', data.shape
    # print 'meandata shape:', meanData.shape
    if fit_point_indices is None:
        log.debug('fitPointIndices shape: None')
    else:
        log.debug('fitPointIndices shape:', fit_point_indices.shape)

    if do_scale:
        align1X0 = numpy.hstack(
            [meanData.mean(0) - data.mean(0), init_rotation, 1.0]
        )
        T1, dataT = alignment_fitting.fitRigidScale(
            data, meanData, align1X0, verbose=verbose
        )
        if landmark_targets is not None:
            landmarkTargetsT = transform3D.transformRigidScale3DAboutP(
                landmark_targets, T1, data.mean(0)
            )
    else:
        align1X0 = numpy.hstack([meanData.mean(0) - data.mean(0), init_rotation])
        # T1, dataT = alignment_fitting.fitRigid( data, meanData, align1X0, verbose=verbose )
        T1, dataT = alignment_fitting.fitRigid(
            data, meanData, align1X0, xtol=1e-6, epsfcn=1e-9, verbose=verbose
        )
        if landmark_targets is not None:
            landmarkTargetsT = transform3D.transformRigid3DAboutP(
                landmark_targets, T1, data.mean(0)
            )

    def _obj(X):
        recon = SSM.reconstruct(
            SSM.getWeightsBySD(fit_modes, X[6:]), fit_modes
        )
        # reconstruct rigid transform
        reconData = transform3D.transformRigid3DAboutCoM(
            # recon.reshape(coord_shape).T, X[:6]
            recon2coords(recon), X[:6]
        )

        # print reconData[466]

        # calc error
        if fit_point_indices is not None:
            E = ((reconData[fit_point_indices, :] - dataT) ** 2.0).sum(1) + \
                mahalanobis(X[6:]) * m_weight
        else:
            E = ((reconData - dataT) ** 2.0).sum(1) + mahalanobis(X[6:]) * m_weight

        if verbose:
            sys.stdout.write('\robj rms:' + str(numpy.sqrt(E.mean())))
            sys.stdout.flush()

        return E

    def _objLandmarks(X):
        recon = SSM.reconstruct(SSM.getWeightsBySD(fit_modes, X[6:]), fit_modes)
        # reconstruct rigid transform
        reconData = transform3D.transformRigid3DAboutCoM(
            # recon.reshape(coord_shape).T, X[:6]
            recon2coords(recon), X[:6]
        )
        # reconLandmarks = transform3D.transformRigid3DAboutP( landmarkEvaluator(recon), X[:6], recon.reshape(coord_shape).T.mean(0) )
        reconLandmarks = landmark_evaluator(reconData.T.ravel())

        # calc error
        if fit_point_indices is not None:
            EData = ((reconData[fit_point_indices, :] - dataT) ** 2.0).sum(1) + \
                    mahalanobis(X[6:]) * m_weight
        else:
            EData = ((reconData - dataT) ** 2.0).sum(1) + mahalanobis(X[6:]) * m_weight

        ELandmarks = ((landmarkTargetsT - reconLandmarks) ** 2.0).sum(1) * landmark_weights
        E = numpy.hstack([EData, ELandmarks])

        if verbose:
            sys.stdout.write(
                '\rPC fit rmse: %6.3f (data: %6.3f) (landmarks: %6.3f)' % \
                (numpy.sqrt(E.mean()), numpy.sqrt(EData.mean()), numpy.sqrt(ELandmarks.mean()))
            )
            sys.stdout.flush()

        return E

    # PC Fit
    x0 = numpy.hstack([[0, 0, 0], init_rotation, numpy.zeros(len(fit_modes), dtype=float)])

    if landmark_targets is None:
        log.debug('using non-landmark obj func')
        xOpt = leastsq(_obj, x0, xtol=1e-6)[0]
    else:
        log.debug('using landmark obj func')
        if verbose:
            log.debug('landmarks')
            log.debug(landmark_targets)
        xOpt = leastsq(_objLandmarks, x0, xtol=1e-6)[0]
    log.debug(' ')

    reconOpt = SSM.reconstruct(
        SSM.getWeightsBySD(fit_modes, xOpt[6:]), fit_modes
    )
    reconDataOpt = transform3D.transformRigid3DAboutCoM(
        # reconOpt.reshape(coord_shape).T, xOpt[:6]
        recon2coords(reconOpt), xOpt[:6]
    )

    # fit has to be done on a subset of recon data is projection is on a subset of variables
    if fit_point_indices is not None:
        # reconDataFit = reconData.ravel()[projectVariables].reshape(coord_shape).T
        reconDataFit = numpy.array(
            # reconDataOpt.T.ravel().reshape(coord_shape).T[fitPointIndices,:]
            reconDataOpt[fit_point_indices, :]
        )
    else:
        reconDataFit = numpy.array(reconDataOpt)

    # TODO
    # inverse transform back to image
    if do_scale:
        align2X0 = numpy.hstack([
            reconDataFit.mean(0) - data.mean(0), -init_rotation, 1.0
        ])
        T2, reconDataFitT = alignment_fitting.fitRigidScale(
            reconDataFit, data, align2X0, verbose=verbose
        )
    else:
        align2X0 = numpy.hstack([
            reconDataFit.mean(0) - data.mean(0), -init_rotation
        ])
        # T2, reconDataT = alignment_fitting.fitRigid( reconData, data, align2X0, verbose=verbose )

        T2, reconDataFitT = alignment_fitting.fitRigid(
            reconDataFit, data, align2X0, xtol=1e-6, epsfcn=1e-9, verbose=verbose
        )

    # transform full recon data to data space if data is a subset of variables
    if fit_point_indices is not None:
        reconDataOptT = transform3D.transformRigid3DAboutCoM(reconDataOpt, T2)
    else:
        reconDataOptT = reconDataFitT

    xOpt[:6] += T2

    return xOpt, reconDataOptT, dataT, reconDataOpt
